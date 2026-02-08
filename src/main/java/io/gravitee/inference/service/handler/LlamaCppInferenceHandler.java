/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.inference.service.handler;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.textgen.InferenceToken;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.llama.cpp.BatchEngine;
import io.gravitee.inference.llama.cpp.EventBusUtils;
import io.gravitee.inference.llama.cpp.ModelConfig;
import io.gravitee.inference.llama.cpp.TagConfig;
import io.gravitee.inference.service.model.LlamaCppModelFactory;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LlamaCppInferenceHandler implements InferenceHandler {

  private static final Logger LOGGER = LoggerFactory.getLogger(LlamaCppInferenceHandler.class);
  private static final long STREAM_ID_TTL_MS = TimeUnit.HOURS.toMillis(1); // 1-hour TTL for orphaned entries

  private final Vertx vertx;
  private final ModelConfig modelConfig;
  private final LlamaCppModelFactory modelFactory;
  private final int key;
  private final AtomicInteger seqIdCounter = new AtomicInteger(0);
  private final Map<Integer, String> streamIds = new ConcurrentHashMap<>();
  private final Map<Integer, Long> streamIdTimestamps = new ConcurrentHashMap<>(); // Track entry ages
  private final ScheduledExecutorService cleanupScheduler = Executors.newScheduledThreadPool(
    1,
    r -> {
      Thread t = new Thread(r, "llama-cpp-stream-cleanup");
      t.setDaemon(true);
      return t;
    }
  );

  private BatchEngine engine;

  public LlamaCppInferenceHandler(
    Vertx vertx,
    ModelConfig modelConfig,
    LlamaCppModelFactory modelFactory,
    int key
  ) {
    this.vertx = vertx;
    this.modelConfig = modelConfig;
    this.modelFactory = modelFactory;
    this.key = key;

    // Start periodic cleanup of orphaned stream IDs (every 5 minutes)
    cleanupScheduler.scheduleAtFixedRate(this::cleanupOrphanedStreamIds, 5, 5, TimeUnit.MINUTES);
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var inferenceRequest = Json.decodeValue(
        message.body(),
        InferenceRequest.class
      );
      InferenceAction action = inferenceRequest.action();
      if (action == null) {
        message.fail(405, "Unsupported action: null");
        return;
      }

      switch (action) {
        case INFER -> handleInfer(message, inferenceRequest.payload());
        case STOP -> handleStop(message, inferenceRequest.payload());
        default -> message.fail(405, "Unsupported action: " + action);
      }
    } catch (Exception e) {
      LOGGER.error("Error handling inference request", e);
      message.fail(400, e.getMessage());
    }
  }

  @Override
  public void loadModel() {
    engine = modelFactory.build(modelConfig, this::publishToken);
  }

  @Override
  public void close() {
    // Shutdown cleanup scheduler
    cleanupScheduler.shutdown();
    try {
      if (!cleanupScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
        cleanupScheduler.shutdownNow();
      }
    } catch (InterruptedException e) {
      cleanupScheduler.shutdownNow();
      Thread.currentThread().interrupt();
    }

    // Close engine
    if (engine != null) {
      engine.close();
    }
  }

  @Override
  public int key() {
    return key;
  }

  private void handleInfer(
    Message<Buffer> message,
    Map<String, Object> payload
  ) {
    ConfigWrapper wrapper = new ConfigWrapper(payload);
    Integer seqId = wrapper.get(Constants.SEQ_ID);
    String streamId = wrapper.get(Constants.MODEL_ADDRESS_KEY);
    if (streamId == null || streamId.isBlank()) {
      message.fail(400, "modelAddress is required");
      return;
    }
    if (seqId == null) {
      seqId = seqIdCounter.incrementAndGet();
    }

    try {
      // Use Phase 2 Request constructor with manual tag parsing for reasoningTags/toolTags
      var baseRequest =
        new io.gravitee.inference.llama.cpp.Request(payload);

      // Parse and attach tags (these aren't in Phase 2 constructor yet)
      TagConfig reasoningTags = payload.get("reasoningTags") != null
        ? getLlamaCppTagConfig(
          (Map<String, Object>) payload.get("reasoningTags")
        )
        : null;
      TagConfig toolTags = payload.get("toolTags") != null
        ? getLlamaCppTagConfig(
          (Map<String, Object>) payload.get("toolTags")
        )
        : null;

      // Reconstruct Request with tags
      io.gravitee.inference.llama.cpp.Request request =
        new io.gravitee.inference.llama.cpp.Request(
          baseRequest.prompt(),
          baseRequest.messages(),
          baseRequest.maxTokens(),
          baseRequest.temperature(),
          baseRequest.topP(),
          baseRequest.presencePenalty(),
          baseRequest.frequencyPenalty(),
          baseRequest.stop(),
          baseRequest.seed(),
          reasoningTags,
          toolTags
        );

      streamIds.put(seqId, streamId);
      streamIdTimestamps.put(seqId, System.currentTimeMillis());
      engine.addSequence(seqId, request);
      message.reply(
        Json.encodeToBuffer(Map.of("status", "started", "seqId", seqId))
      );
    } catch (Exception e) {
      LOGGER.error("Error handling infer request for seqId {}", seqId, e);
      streamIds.remove(seqId);
      streamIdTimestamps.remove(seqId);
      message.fail(400, e.getMessage());
    }
  }

  private void handleStop(
    Message<Buffer> message,
    Map<String, Object> payload
  ) {
    ConfigWrapper wrapper = new ConfigWrapper(payload);
    Integer seqId = wrapper.get(Constants.SEQ_ID);
    if (seqId == null) {
      message.fail(400, "seqId is required");
      return;
    }

    var token = engine.cancelSequence(seqId);
    if (token != null) {
      publishToken(token);
    }
    streamIds.remove(seqId);
    streamIdTimestamps.remove(seqId);
    message.reply(
      Json.encodeToBuffer(Map.of("status", "cancelled", "seqId", seqId))
    );
  }

  private void publishToken(InferenceToken<String> token) {
    String streamId = streamIds.get(token.seqId());
    if (streamId == null) {
      return;
    }
    JsonObject payload = new JsonObject();
    payload.put("seqId", token.seqId());
    payload.put("token", token.token());
    payload.put("index", token.index());
    payload.put("isFinal", token.isFinal());
    payload.put("finishReason", token.finishReason());
    payload.put("promptTokens", token.promptTokens());
    payload.put("completionTokens", token.completionTokens());
    payload.put("reasoningTokens", token.reasoningTokens());
    payload.put("toolTokens", token.toolTokens());

    // Optimized: Use Map directly instead of creating new JsonObject with 14 put() calls
    var performance = token.performance();
    if (performance != null) {
      Map<String, Object> perfMap = Map.ofEntries(
        Map.entry("startTimeMs", performance.startTimeMs()),
        Map.entry("loadTimeMs", performance.loadTimeMs()),
        Map.entry("promptEvalTimeMs", performance.promptEvalTimeMs()),
        Map.entry("evalTimeMs", performance.evalTimeMs()),
        Map.entry(
          "promptTokensEvaluated",
          performance.promptTokensEvaluated()
        ),
        Map.entry("tokensGenerated", performance.tokensGenerated()),
        Map.entry("tokensReused", performance.tokensReused()),
        Map.entry("samplingTimeMs", performance.samplingTimeMs()),
        Map.entry("sampleCount", performance.sampleCount()),
        Map.entry(
          "promptTokensPerSecond",
          performance.promptTokensPerSecond()
        ),
        Map.entry(
          "generationTokensPerSecond",
          performance.generationTokensPerSecond()
        ),
        Map.entry(
          "totalProcessingTimeMs",
          performance.totalProcessingTimeMs()
        ),
        Map.entry(
          "averageSamplingTimeMs",
          performance.averageSamplingTimeMs()
        )
      );
      payload.put("performance", perfMap);
    }
    vertx
      .eventBus()
      .publish(EventBusUtils.tokensAddress(streamId, token.seqId()), payload);
    if (token.isFinal()) {
      streamIds.remove(token.seqId());
      streamIdTimestamps.remove(token.seqId()); // Clean timestamp on completion
    }
  }

  private static TagConfig getLlamaCppTagConfig(Map<String, Object> tags) {
    return new TagConfig(
      String.valueOf(tags.get("openToken")),
      String.valueOf(tags.get("closeToken"))
    );
  }

  @SuppressWarnings("unchecked")
  private List<io.gravitee.inference.api.textgen.Content> parseMediaContent(
    Object value
  ) {
    if (!(value instanceof List<?> list)) {
      return List.of();
    }
    List<io.gravitee.inference.api.textgen.Content> media = new ArrayList<>();
    for (Object item : list) {
      if (item instanceof Map<?, ?> map) {
        String type = stringValue(map.get("type"));
        String data = stringValue(map.get("data"));
        String mediaTypeStr = stringValue(map.get("mediaType"));
        if (type != null && data != null) {
          io.gravitee.inference.api.textgen.MediaType mediaType =
            mediaTypeStr != null
              ? io.gravitee.inference.api.textgen.MediaType.fromString(
                mediaTypeStr
              )
              : io.gravitee.inference.api.textgen.MediaType
                .APPLICATION_OCTET_STREAM;
          if ("image".equals(type)) {
            media.add(
              new io.gravitee.inference.api.textgen.ImageContent(
                mediaType,
                data
              )
            );
          } else if ("audio".equals(type)) {
            media.add(
              new io.gravitee.inference.api.textgen.AudioContent(
                mediaType,
                data
              )
            );
          }
        }
      }
    }
    return media;
  }

  private List<String> parseStop(Object value) {
    if (value instanceof String s) {
      return List.of(s);
    }
    if (value instanceof List<?> list) {
      List<String> stops = new ArrayList<>();
      for (Object item : list) {
        String stop = stringValue(item);
        if (stop != null) {
          stops.add(stop);
        }
      }
      return stops.isEmpty() ? null : stops;
    }
    return null;
  }

  private io.gravitee.inference.api.textgen.Role toRole(String role) {
    if (role == null) {
      return io.gravitee.inference.api.textgen.Role.USER;
    }

    return switch (role.toLowerCase().trim()) {
      case "assistant" -> io.gravitee.inference.api.textgen.Role.ASSISTANT;
      case "system" -> io.gravitee.inference.api.textgen.Role.SYSTEM;
      case "user" -> io.gravitee.inference.api.textgen.Role.USER;
      default -> {
        LOGGER.warn("Unknown role '{}', defaulting to USER", role);
        yield io.gravitee.inference.api.textgen.Role.USER;
      }
    };
  }

  private String stringValue(Object value) {
    return value == null ? null : value.toString();
  }

  private Integer intValue(Object value, Integer defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number number) {
      return number.intValue();
    }
    return Integer.parseInt(value.toString());
  }

  private Float floatValue(Object value, Float defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number number) {
      return number.floatValue();
    }
    return Float.parseFloat(value.toString());
  }

  /**
   * Cleanup helper method for removing orphaned stream IDs that have exceeded the TTL.
   * Runs periodically (every 5 minutes) to prevent memory leaks from orphaned entries.
   * This addresses the critical memory leak issue where entries persisted indefinitely
   * if sequences timeout or error without explicit STOP.
   */
  private void cleanupOrphanedStreamIds() {
    long now = System.currentTimeMillis();
    long cutoff = now - STREAM_ID_TTL_MS;

    streamIdTimestamps
      .entrySet()
      .removeIf(entry -> {
        int seqId = entry.getKey();
        long timestamp = entry.getValue();

        if (timestamp < cutoff) {
          LOGGER.warn("Cleaning up orphaned stream ID {} (age: {}ms)", seqId, now - timestamp);
          streamIds.remove(seqId);
          return true;
        }
        return false;
      });
  }
}

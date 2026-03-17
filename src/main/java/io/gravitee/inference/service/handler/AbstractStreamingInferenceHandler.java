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
import io.gravitee.inference.api.textgen.AbstractBatchEngine;
import io.gravitee.inference.api.textgen.GenerationRequest;
import io.gravitee.inference.api.textgen.InferenceToken;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for streaming text-generation inference handlers.
 *
 * <p>Encapsulates the common lifecycle (handle → infer/stop, token publishing,
 * stream-ID cleanup) shared by all batch-engine-backed handlers (llama.cpp, vLLM).
 *
 * <p>Subclasses only need to implement:
 * <ul>
 *   <li>{@link #buildEngine()} — construct and start the engine</li>
 *   <li>{@link #buildRequest(Map)} — translate the raw payload into the engine-specific request</li>
 *   <li>{@link #tokensAddress(String, int)} — compute the event-bus publish address for tokens</li>
 * </ul>
 *
 * @param <R> the engine-specific request type (e.g. {@code Request}, {@code VllmRequest})
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class AbstractStreamingInferenceHandler<
  R extends GenerationRequest
>
  implements InferenceHandler {

  private static final long STREAM_ID_TTL_MS = TimeUnit.HOURS.toMillis(1);

  protected final Logger logger = LoggerFactory.getLogger(getClass());
  protected final Vertx vertx;
  private final int key;
  private final AtomicInteger seqIdCounter = new AtomicInteger(0);
  private final Map<Integer, String> streamIds = new ConcurrentHashMap<>();
  private final Map<Integer, Long> streamIdTimestamps =
    new ConcurrentHashMap<>();
  private final ScheduledExecutorService cleanupScheduler;

  protected AbstractBatchEngine<?, R, String, ?> engine;

  protected AbstractStreamingInferenceHandler(
    Vertx vertx,
    int key,
    String cleanupThreadName
  ) {
    this.vertx = vertx;
    this.key = key;
    this.cleanupScheduler = Executors.newScheduledThreadPool(1, r -> {
      Thread t = new Thread(r, cleanupThreadName);
      t.setDaemon(true);
      return t;
    });
    cleanupScheduler.scheduleAtFixedRate(
      this::cleanupOrphanedStreamIds,
      5,
      5,
      TimeUnit.MINUTES
    );
  }

  // ── Abstract hooks ──────────────────────────────────────────────────────

  /** Build and start the engine. Called once by {@link #loadModel()}. */
  protected abstract AbstractBatchEngine<?, R, String, ?> buildEngine();

  /**
   * Translate a raw event-bus payload into the engine-specific request type.
   *
   * @param payload the decoded payload map from the inference request
   * @return the engine-specific request
   */
  protected abstract R buildRequest(Map<String, Object> payload);

  /**
   * Compute the event-bus address to publish tokens on.
   *
   * @param streamId the stream identifier (model address key)
   * @param seqId    the sequence identifier
   * @return the fully-qualified event-bus address
   */
  protected abstract String tokensAddress(String streamId, int seqId);

  // ── InferenceHandler implementation ──────────────────────────────────────

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
      logger.error("Error handling inference request", e);
      message.fail(400, e.getMessage());
    }
  }

  @Override
  public void loadModel() {
    engine = buildEngine();
  }

  @Override
  public void close() {
    cleanupScheduler.shutdown();
    try {
      if (!cleanupScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
        cleanupScheduler.shutdownNow();
      }
    } catch (InterruptedException e) {
      cleanupScheduler.shutdownNow();
      Thread.currentThread().interrupt();
    }

    if (engine != null) {
      engine.close();
    }
  }

  @Override
  public int key() {
    return key;
  }

  // ── Private dispatch ──────────────────────────────────────────────────

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
      R request = buildRequest(payload);

      streamIds.put(seqId, streamId);
      streamIdTimestamps.put(seqId, System.currentTimeMillis());
      engine.addSequence(seqId, request);
      message.reply(
        Json.encodeToBuffer(Map.of("status", "started", "seqId", seqId))
      );
    } catch (Exception e) {
      logger.error("Error handling infer request for seqId {}", seqId, e);
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

  // ── Token publishing ──────────────────────────────────────────────────

  protected void publishToken(InferenceToken<String> token) {
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

    var performance = token.performance();
    if (performance != null) {
      Map<String, Object> perfMap = Map.ofEntries(
        Map.entry("startTimeMs", performance.startTimeMs()),
        Map.entry("loadTimeMs", performance.loadTimeMs()),
        Map.entry("promptEvalTimeMs", performance.promptEvalTimeMs()),
        Map.entry("evalTimeMs", performance.evalTimeMs()),
        Map.entry("promptTokensEvaluated", performance.promptTokensEvaluated()),
        Map.entry("tokensGenerated", performance.tokensGenerated()),
        Map.entry("tokensReused", performance.tokensReused()),
        Map.entry("samplingTimeMs", performance.samplingTimeMs()),
        Map.entry("sampleCount", performance.sampleCount()),
        Map.entry("promptTokensPerSecond", performance.promptTokensPerSecond()),
        Map.entry(
          "generationTokensPerSecond",
          performance.generationTokensPerSecond()
        ),
        Map.entry("totalProcessingTimeMs", performance.totalProcessingTimeMs()),
        Map.entry("averageSamplingTimeMs", performance.averageSamplingTimeMs())
      );
      payload.put("performance", perfMap);
    }
    vertx.eventBus().publish(tokensAddress(streamId, token.seqId()), payload);
    if (token.isFinal()) {
      streamIds.remove(token.seqId());
      streamIdTimestamps.remove(token.seqId());
    }
  }

  // ── Tag parsing helper ──────────────────────────────────────────────────

  protected static <T> T parseTagConfig(
    Map<String, Object> tags,
    java.util.function.BiFunction<String, String, T> factory
  ) {
    return factory.apply(
      String.valueOf(tags.get("openToken")),
      String.valueOf(tags.get("closeToken"))
    );
  }

  // ── Orphaned stream cleanup ──────────────────────────────────────────

  private void cleanupOrphanedStreamIds() {
    long now = System.currentTimeMillis();
    long cutoff = now - STREAM_ID_TTL_MS;

    streamIdTimestamps
      .entrySet()
      .removeIf(entry -> {
        int seqId = entry.getKey();
        long timestamp = entry.getValue();

        if (timestamp < cutoff) {
          logger.warn(
            "Cleaning up orphaned stream ID {} (age: {}ms)",
            seqId,
            now - timestamp
          );
          streamIds.remove(seqId);
          return true;
        }
        return false;
      });
  }
}

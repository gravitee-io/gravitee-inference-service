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
import io.gravitee.inference.llama.cpp.*;
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
import java.util.concurrent.atomic.AtomicInteger;

public class LlamaCppInferenceHandler implements InferenceHandler {

  private final Vertx vertx;
  private final ModelConfig modelConfig;
  private final LlamaCppModelFactory modelFactory;
  private final int key;
  private final AtomicInteger seqIdCounter = new AtomicInteger(0);
  private final Map<Integer, String> streamIds = new ConcurrentHashMap<>();

  private BatchEngine engine;

  public LlamaCppInferenceHandler(Vertx vertx, ModelConfig modelConfig, LlamaCppModelFactory modelFactory, int key) {
    this.vertx = vertx;
    this.modelConfig = modelConfig;
    this.modelFactory = modelFactory;
    this.key = key;
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var inferenceRequest = Json.decodeValue(message.body(), InferenceRequest.class);
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
      message.fail(400, e.getMessage());
    }
  }

  @Override
  public void loadModel() {
    engine = modelFactory.build(modelConfig, this::publishToken);
  }

  @Override
  public void close() {
    if (engine != null) {
      engine.close();
    }
  }

  @Override
  public int key() {
    return key;
  }

  private void handleInfer(Message<Buffer> message, Map<String, Object> payload) {
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

    Request request = parseGenerationRequest(payload);
    streamIds.put(seqId, streamId);
    engine.addSequence(seqId, request);
    message.reply(Json.encodeToBuffer(Map.of("status", "started", "seqId", seqId)));
  }

  private void handleStop(Message<Buffer> message, Map<String, Object> payload) {
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
    message.reply(Json.encodeToBuffer(Map.of("status", "cancelled", "seqId", seqId)));
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
    var performance = token.performance();
    if (performance != null) {
      JsonObject perf = new JsonObject();
      perf.put("startTimeMs", performance.startTimeMs());
      perf.put("loadTimeMs", performance.loadTimeMs());
      perf.put("promptEvalTimeMs", performance.promptEvalTimeMs());
      perf.put("evalTimeMs", performance.evalTimeMs());
      perf.put("promptTokensEvaluated", performance.promptTokensEvaluated());
      perf.put("tokensGenerated", performance.tokensGenerated());
      perf.put("tokensReused", performance.tokensReused());
      perf.put("samplingTimeMs", performance.samplingTimeMs());
      perf.put("sampleCount", performance.sampleCount());
      perf.put("promptTokensPerSecond", performance.promptTokensPerSecond());
      perf.put("generationTokensPerSecond", performance.generationTokensPerSecond());
      perf.put("totalProcessingTimeMs", performance.totalProcessingTimeMs());
      perf.put("averageSamplingTimeMs", performance.averageSamplingTimeMs());
      payload.put("performance", perf);
    }
    vertx.eventBus().publish(EventBusUtils.tokensAddress(streamId, token.seqId()), payload);
    if (token.isFinal()) {
      streamIds.remove(token.seqId());
    }
  }

  private Request parseGenerationRequest(Map<String, Object> payload) {
    String prompt = stringValue(payload.get(Constants.PROMPT));
    List<io.gravitee.inference.api.textgen.ChatMessage> messages = parseMessages(payload.get(Constants.MESSAGES));

    return new Request(
      prompt,
      messages,
      intValue(payload.get(Constants.MAX_TOKENS), null),
      floatValue(payload.get(Constants.TEMPERATURE), null),
      floatValue(payload.get(Constants.TOP_P), null),
      floatValue(payload.get(Constants.PRESENCE_PENALTY), null),
      floatValue(payload.get(Constants.FREQUENCY_PENALTY), null),
      parseStop(payload.get(Constants.STOP)),
      intValue(payload.get(Constants.SEED), 42),
      payload.get("reasoningTags") != null ? getLlamaCppTagConfig((Map<String, Object>) payload.get("reasoningTags")) : null,
      payload.get("toolTags") != null ? getLlamaCppTagConfig((Map<String, Object>) payload.get("toolTags")) : null
    );
  }

  private static TagConfig getLlamaCppTagConfig(Map<String, Object> tags) {
    return new TagConfig(String.valueOf(tags.get("openTag")), String.valueOf(tags.get("endTag")));
  }

  private List<io.gravitee.inference.api.textgen.ChatMessage> parseMessages(Object value) {
    if (!(value instanceof List<?> list)) {
      return null;
    }
    List<io.gravitee.inference.api.textgen.ChatMessage> result = new ArrayList<>();
    for (Object item : list) {
      if (item instanceof Map<?, ?> map) {
        String role = stringValue(map.get("role"));
        String content = stringValue(map.get("content"));
        if (role != null && content != null) {
          result.add(new io.gravitee.inference.api.textgen.ChatMessage(toRole(role), content));
        }
      }
    }
    return result;
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
    return switch (role) {
      case "assistant" -> io.gravitee.inference.api.textgen.Role.ASSISTANT;
      case "system" -> io.gravitee.inference.api.textgen.Role.SYSTEM;
      default -> io.gravitee.inference.api.textgen.Role.USER;
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
}

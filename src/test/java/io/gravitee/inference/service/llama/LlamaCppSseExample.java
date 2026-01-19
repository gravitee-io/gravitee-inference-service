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
package io.gravitee.inference.service.llama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.gravitee.common.utils.TimeProvider;
import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.llama.cpp.EventBusUtils;
import io.gravitee.inference.service.handler.ModelHandler;
import io.gravitee.inference.service.provider.ModelProviderRegistry;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.vertx.core.Vertx;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.core.json.JsonObject;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Runnable SSE example for llama.cpp streaming.
 *
 * Usage:
 *   java io.gravitee.inference.service.llama.LlamaCppSseExample <modelName> <modelPath>
 */
public final class LlamaCppSseExample {

  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  public static void main(String[] args) {
    if (args.length < 2) {
      System.err.println("Usage: LlamaCppSseExample <modelName> <modelPath>");
      System.exit(1);
    }

    String modelName = args[0];
    String modelPath = args[1];

    Vertx vertx = Vertx.vertx();
    io.vertx.rxjava3.core.Vertx rxVertx = io.vertx.rxjava3.core.Vertx.newInstance(vertx);
    var modelHandler = new ModelHandler(rxVertx, new HandlerRepository(), new ModelProviderRegistry(rxVertx, "."));
    rxVertx.eventBus().<Buffer>consumer(Constants.SERVICE_INFERENCE_MODELS_ADDRESS).handler(modelHandler);

    String address = loadModel(vertx, modelName, modelPath);

    vertx
      .createHttpServer()
      .requestHandler(request -> {
        if (!"/v1/chat/completions".equals(request.path())) {
          request.response().setStatusCode(404).end();
          return;
        }

        request.bodyHandler(body -> {
          try {
            JsonNode payload = OBJECT_MAPPER.readTree(body.toString());
            boolean stream = payload.at("/stream").asBoolean(false);
            if (!stream) {
              request.response().setStatusCode(400).end("stream=true is required for this example");
              return;
            }

            Map<String, Object> inferPayload = new HashMap<>();
            inferPayload.put(Constants.MODEL_NAME, modelName);
            inferPayload.put(Constants.MODEL_ADDRESS_KEY, address);
            inferPayload.put(
              Constants.MESSAGES,
              List.of(Map.of("role", "user", "content", payload.at("/messages/0/content").asText("")))
            );

            var inferenceRequest = new InferenceRequest(InferenceAction.INFER, inferPayload);

            vertx
              .eventBus()
              .request(address, Json.encodeToBuffer(inferenceRequest))
              .onSuccess(reply -> {
                JsonObject json = new JsonObject(((Buffer) reply.body()).toString());
                int seqId = json.getInteger("seqId");
                streamTokens(vertx, request, modelName, address, seqId);
              })
              .onFailure(error -> request.response().setStatusCode(500).end(error.getMessage()));
          } catch (Exception e) {
            request.response().setStatusCode(400).end("Invalid JSON payload");
          }
        });
      })
      .listen(8090);

    System.out.println("SSE example running on http://localhost:8090/v1/chat/completions");
  }

  private static String loadModel(Vertx vertx, String modelName, String modelPath) {
    Map<String, Object> payload = new HashMap<>();
    payload.put(Constants.INFERENCE_FORMAT, InferenceFormat.LLAMA_CPP.name());
    payload.put(Constants.INFERENCE_TYPE, InferenceType.TEXT_GENERATION.name());
    payload.put(Constants.MODEL_NAME, modelName);
    payload.put(Constants.MODEL_PATH, modelPath);

    var request = new InferenceRequest(InferenceAction.START, payload);
    var reply = vertx
      .eventBus()
      .request(Constants.SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(request))
      .toCompletionStage()
      .toCompletableFuture()
      .join();
    return ((Buffer) reply.body()).toString();
  }

  private static void streamTokens(
    Vertx vertx,
    io.vertx.core.http.HttpServerRequest request,
    String modelName,
    String address,
    int seqId
  ) {
    var response = request.response();
    response.putHeader("Content-Type", "text/event-stream");

    String tokensAddress = EventBusUtils.tokensAddress(address, seqId);
    var consumer = vertx.eventBus().consumer(tokensAddress);

    long created = TimeProvider.instantNow().getEpochSecond();
    String responseId = "chatcmpl-" + created;

    consumer.handler(message -> {
      JsonObject token = (JsonObject) message.body();
      boolean isFinal = token.getBoolean("isFinal", false);

      ObjectNode chunk = OBJECT_MAPPER.createObjectNode();
      chunk.put("id", responseId);
      chunk.put("object", "chat.completion.chunk");
      chunk.put("created", created);
      chunk.put("model", modelName);
      ArrayNode choices = chunk.putArray("choices");
      ObjectNode choice = choices.addObject();
      choice.put("index", 0);
      ObjectNode delta = choice.putObject("delta");
      delta.put("content", token.getString("token", ""));
      if (isFinal) {
        choice.put("finish_reason", token.getString("finishReason", "stop"));
      } else {
        choice.putNull("finish_reason");
      }

      response.write("data: " + chunk + "\n\n");
      if (isFinal) {
        response.write("data: [DONE]\n\n");
        response.end();
        consumer.unregister();
      }
    });
  }
}

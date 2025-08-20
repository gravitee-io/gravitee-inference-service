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

import static io.gravitee.inference.api.Constants.*;

import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.provider.ModelProviderRegistry;
import io.gravitee.inference.service.repository.ModelRepository;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ModelHandler implements Handler<Message<Buffer>> {

  private final Logger log = LoggerFactory.getLogger(ModelHandler.class);

  private final Vertx vertx;
  private final ModelRepository repository;
  private final Map<String, InferenceHandler<?>> inferenceHandlers = new ConcurrentHashMap<>();
  private final ModelProviderRegistry modelProviderRegistry;

  public ModelHandler(Vertx vertx, ModelRepository repository, ModelProviderRegistry modelProviderRegistry) {
    this.vertx = vertx;
    this.repository = repository;
    this.modelProviderRegistry = modelProviderRegistry;
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var inferenceRequest = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (inferenceRequest.action()) {
        case START -> handleStart(message, inferenceRequest);
        case STOP -> handleStop(message, inferenceRequest);
        case null, default -> message.fail(405, "Unsupported action: " + inferenceRequest.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  private void handleStart(Message<Buffer> message, InferenceRequest inferenceRequest) {
    InferenceFormat inferenceFormat = InferenceFormat.valueOf(inferenceRequest.payload().get(INFERENCE_FORMAT).toString());
    String address = String.format(SERVICE_INFERENCE_MODELS_INFER_TEMPLATE, UUID.randomUUID());

    var handler =
      switch (inferenceFormat) {
        case ONNX_BERT -> new LocalInferenceHandler(address, vertx);
        case OPENAI, HTTP -> new RemoteInferenceHandler(address, vertx);
      };

    inferenceHandlers.put(address, handler);

    System.out.println("Starting inference handler for " + inferenceFormat);

    modelProviderRegistry
      .getProvider(inferenceFormat)
      .loadModel(inferenceRequest, repository)
      .subscribe(
        handler::setModel,
        error -> {
          log.error("Failed to start inference handler", error);
          inferenceHandlers.remove(address);
        }
      );

    message.reply(Buffer.buffer(address));
  }

  private void handleStop(Message<Buffer> message, InferenceRequest inferenceRequest) {
    var config = new ConfigWrapper(inferenceRequest.payload());
    var address = config.<String>get(MODEL_ADDRESS_KEY);

    if (inferenceHandlers.containsKey(address)) {
      var inferenceHandler = inferenceHandlers.remove(address);
      inferenceHandler.close();
      repository.remove(inferenceHandler.getModel());
      message.reply(Buffer.buffer(address));
    } else {
      throw new IllegalArgumentException("Could not find inference handler for address: " + address);
    }
  }

  public void close() {
    inferenceHandlers.forEach((__, handler) -> handler.close());
  }
}

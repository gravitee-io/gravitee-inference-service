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

import static io.gravitee.inference.api.Constants.SERVICE_INFERENCE_MODELS_CREATE;
import static io.gravitee.inference.api.Constants.SERVICE_INFERENCE_MODELS_INFER_TEMPLATE;

import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.handler.action.CreateModelAction;
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class InferenceCrudHandler implements Handler<Message<Buffer>> {

  private final Logger LOGGER = LoggerFactory.getLogger(InferenceCrudHandler.class);

  private final Vertx vertx;
  private final Map<String, InferenceHandler> inferenceHandlers = new ConcurrentHashMap<>();
  private final Disposable consumer;
  private final CreateModelAction action;

  public InferenceCrudHandler(Vertx vertx, CreateModelAction action) {
    this.vertx = vertx;
    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(SERVICE_INFERENCE_MODELS_CREATE)
        .toObservable()
        .subscribe(this::handle, throwable -> LOGGER.error("Inference service handler failed", throwable));
    this.action = action;
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var inferenceRequest = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (inferenceRequest.action()) {
        case CREATE -> {
          var model = action.handle(inferenceRequest);
          var address = String.format(SERVICE_INFERENCE_MODELS_INFER_TEMPLATE, UUID.randomUUID());
          inferenceHandlers.put(address, new InferenceHandler(address, model, vertx));
          message.reply(Buffer.buffer(address));
        }
        case null, default -> message.fail(405, "Unsupported action: " + inferenceRequest.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  public void close() {
    if (!consumer.isDisposed()) {
      consumer.dispose();
    }
    inferenceHandlers.forEach((__, handler) -> handler.close());
  }
}

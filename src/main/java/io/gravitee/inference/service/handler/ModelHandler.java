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

import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.repository.ModelRepository;
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.RxHelper;
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
public class ModelHandler implements Handler<Message<Buffer>> {

  private final Logger LOGGER = LoggerFactory.getLogger(ModelHandler.class);

  private final Vertx vertx;
  private final ModelRepository repository;

  private final Map<String, InferenceHandler> inferenceHandlers = new ConcurrentHashMap<>();
  private final Disposable consumer;

  public ModelHandler(Vertx vertx, ModelRepository repository) {
    this.vertx = vertx;
    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(SERVICE_INFERENCE_MODELS_ADDRESS)
        .toObservable()
        .subscribeOn(RxHelper.blockingScheduler(vertx.getDelegate()))
        .observeOn(RxHelper.blockingScheduler(vertx.getDelegate()))
        .subscribe(this::handle, throwable -> LOGGER.error("Inference service handler failed", throwable));
    this.repository = repository;
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var inferenceRequest = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (inferenceRequest.action()) {
        case START -> {
          var model = repository.add(inferenceRequest);
          var address = String.format(SERVICE_INFERENCE_MODELS_INFER_TEMPLATE, UUID.randomUUID());
          inferenceHandlers.put(address, new InferenceHandler(address, model, vertx));
          message.reply(Buffer.buffer(address));
        }
        case STOP -> {
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

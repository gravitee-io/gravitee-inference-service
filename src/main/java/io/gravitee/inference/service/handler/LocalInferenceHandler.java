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
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.repository.Model;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class LocalInferenceHandler implements Handler<Message<Buffer>>, InferenceHandler<Object> {

  private final Logger LOGGER = LoggerFactory.getLogger(LocalInferenceHandler.class);

  private final AtomicReference<Model<Object>> model = new AtomicReference<>();
  private final Disposable consumer;
  private final String address;

  public LocalInferenceHandler(String address, Vertx vertx) {
    this.address = address;
    LOGGER.debug("Starting Inference handler at {}", address);
    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(address)
        .toObservable()
        .subscribeOn(RxHelper.blockingScheduler(vertx))
        .observeOn(RxHelper.blockingScheduler(vertx))
        .subscribe(this::handle, throwable -> LOGGER.error("Inference service handler failed", throwable));
    LOGGER.debug("Inference handler at {} started", address);
  }

  @Override
  public void handle(Message<Buffer> message) {
    if (model.get() == null) {
      message.fail(503, "Model is not ready");
      return;
    }

    try {
      var request = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (request.action()) {
        case INFER -> {
          var config = new ConfigWrapper(request.payload());
          var loadedModel = model.get().inferenceModel();
          var output = loadedModel.infer(config.get(Constants.INPUT));
          if (output instanceof Maybe<?> single) {
            single.subscribe(o -> message.reply(Json.encodeToBuffer(o)));
          } else {
            message.reply(Json.encodeToBuffer(output));
          }
        }
        case null, default -> message.fail(405, "Unsupported action: " + request.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  public Model<Object> getModel() {
    return model.get();
  }

  public void close() {
    if (!consumer.isDisposed()) {
      LOGGER.debug("Stopping Inference handler: {}", address);
      consumer.dispose();
      LOGGER.debug("Inference handler {} stopped", address);
    }
  }

  public void setModel(Model<Object> model) {
    this.model.set(model);
  }
}

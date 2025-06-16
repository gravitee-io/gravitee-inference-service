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
import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.repository.Model;
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class InferenceHandler implements Handler<Message<Buffer>> {

  private final Logger LOGGER = LoggerFactory.getLogger(InferenceHandler.class);

  private final Model model;
  private final Disposable consumer;
  private final String address;

  public InferenceHandler(String address, Model model, Vertx vertx) {
    this.model = model;
    this.address = address;
    LOGGER.debug("Starting Inference handler at {}", address);
    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(address)
        .toObservable()
        .subscribeOn(RxHelper.blockingScheduler(vertx.getDelegate()))
        .observeOn(RxHelper.blockingScheduler(vertx.getDelegate()))
        .subscribe(this::handle, throwable -> LOGGER.error("Inference service handler failed", throwable));
    LOGGER.debug("Inference handler at {} started", address);
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var request = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (request.action()) {
        case INFER -> {
          var config = new ConfigWrapper(request.payload());
          message.reply(Json.encodeToBuffer(model.inferenceModel().infer(config.get(Constants.INPUT))));
        }
        case null, default -> message.fail(405, "Unsupported action: " + request.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  public Model getModel() {
    return model;
  }

  public void close() {
    if (!consumer.isDisposed()) {
      LOGGER.debug("Stopping Inference handler: {}", address);
      consumer.dispose();
      LOGGER.debug("Inference handler {} stopped", address);
    }
  }
}

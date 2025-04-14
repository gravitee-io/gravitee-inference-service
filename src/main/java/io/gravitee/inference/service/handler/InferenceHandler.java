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
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class InferenceHandler implements Handler<Message<Buffer>> {

  private final InferenceModel<?, String, ?> model;
  private final Disposable consumer;
  private final Logger LOGGER = LoggerFactory.getLogger(InferenceHandler.class);

  public InferenceHandler(String address, InferenceModel<?, String, ?> model, Vertx vertx) {
    this.model = model;
    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(address)
        .toObservable()
        .subscribe(this::handle, throwable -> LOGGER.error("Inference service handler failed", throwable));
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var request = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (request.action()) {
        case INFER -> {
          var config = new ConfigWrapper(request.payload());
          message.reply(Json.encodeToBuffer(model.infer(config.get(Constants.INPUT))));
        }
        case null, default -> message.fail(405, "Unsupported action: " + request.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  public void close() {
    if (!consumer.isDisposed()) {
      consumer.dispose();
    }
  }
}

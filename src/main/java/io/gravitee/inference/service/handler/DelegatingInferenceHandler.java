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

import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.buffer.Buffer;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class DelegatingInferenceHandler implements InferenceHandler {

  private static final Logger LOGGER = LoggerFactory.getLogger(DelegatingInferenceHandler.class);
  private final String address;
  private final Disposable consumer;
  private InferenceHandler delegate;

  public DelegatingInferenceHandler(String address, Vertx vertx, boolean isRemote) {
    this.address = address;
    LOGGER.debug("Starting Delegating Inference handler at {}", address);
    var scheduler = isRemote ? RxHelper.scheduler(vertx) : RxHelper.blockingScheduler(vertx);

    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(address)
        .toObservable()
        .subscribeOn(scheduler)
        .observeOn(scheduler)
        .subscribe(this::handle);

    LOGGER.debug("Inference handler at {} started", address);
  }

  @Override
  public void handle(Message<Buffer> message) {
    if (delegate == null) {
      message.fail(503, "Model is not ready");
      return;
    }
    delegate.handle(message);
  }

  public InferenceHandler getDelegate() {
    return delegate;
  }

  public void setDelegate(InferenceHandler handle) {
    this.delegate = handle;
  }

  public void close() {
    if (!consumer.isDisposed()) {
      LOGGER.debug("Stopping DelegatingInferenceHandler: {}", address);
      consumer.dispose();
      LOGGER.debug("DelegatingInferenceHandler handler {} stopped", address);
    }
  }
}

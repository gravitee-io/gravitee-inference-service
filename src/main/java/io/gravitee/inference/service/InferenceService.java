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
package io.gravitee.inference.service;

import static io.gravitee.inference.api.Constants.SERVICE_INFERENCE_MODELS_ADDRESS;

import io.gravitee.common.service.AbstractService;
import io.gravitee.inference.service.handler.ModelHandler;
import io.gravitee.inference.service.provider.ModelProviderRegistry;
import io.gravitee.inference.service.repository.ModelRepository;
import io.gravitee.reactive.webclient.huggingface.downloader.HuggingFaceDownloader;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.buffer.Buffer;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class InferenceService extends AbstractService<InferenceService> {

  private static final String INFERENCE_SERVICE = "Gravitee Inference - Service";

  private final Logger LOGGER = LoggerFactory.getLogger(InferenceService.class);
  private final Vertx vertx;
  private final String modelPath;

  private ModelHandler crudHandler;

  @NonNull
  private Disposable consumer;

  @Autowired
  public InferenceService(
    Vertx vertx,
    @Value("${inference.path:#{systemProperties['gravitee.home']}/models}") String modelPath
  ) {
    this.vertx = vertx;
    this.modelPath = modelPath;
  }

  @Override
  protected String name() {
    return INFERENCE_SERVICE;
  }

  @Override
  protected void doStart() throws Exception {
    LOGGER.debug("Starting Inference service");
    super.doStart();
    crudHandler = new ModelHandler(vertx, new ModelRepository(vertx), new ModelProviderRegistry(vertx, modelPath));
    consumer =
      vertx
        .eventBus()
        .<Buffer>consumer(SERVICE_INFERENCE_MODELS_ADDRESS)
        .toObservable()
        .subscribe(crudHandler::handle, throwable -> LOGGER.error("Inference service handler failed", throwable));
  }

  @Override
  protected void doStop() throws Exception {
    LOGGER.debug("Stopping Inference service");
    super.doStop();
    crudHandler.close();
    consumer.dispose();
  }
}

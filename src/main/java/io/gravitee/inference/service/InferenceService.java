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
import io.gravitee.inference.service.repository.HandlerRepository;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.disposables.Disposable;
import io.vertx.core.buffer.Buffer;
import io.vertx.rxjava3.ContextScheduler;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.WorkerExecutor;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@Service
public class InferenceService extends AbstractService<InferenceService> {

  private static final String INFERENCE_SERVICE =
    "Gravitee Inference - Service";
  static final String MODEL_LOADER_POOL_NAME = "gio-inference-model-loader";
  static final int DEFAULT_MODEL_LOADER_POOL_SIZE = 2;
  static final long DEFAULT_MODEL_LOADER_MAX_EXECUTE_TIME_MS = 300_000L;

  private final Logger LOGGER = LoggerFactory.getLogger(InferenceService.class);
  private final Vertx vertx;
  private final String modelPath;
  private final int modelLoaderPoolSize;
  private final long modelLoaderMaxExecuteTimeMs;

  private ModelHandler crudHandler;
  private WorkerExecutor modelLoader;

  @NonNull
  private Disposable consumer;

  public InferenceService(Vertx vertx, String modelPath) {
    this(
      vertx,
      modelPath,
      DEFAULT_MODEL_LOADER_POOL_SIZE,
      DEFAULT_MODEL_LOADER_MAX_EXECUTE_TIME_MS
    );
  }

  @Autowired
  public InferenceService(
    Vertx vertx,
    @Value(
      "${inference.path:#{systemProperties['gravitee.home']}/models}"
    ) String modelPath,
    @Value(
      "${inference.models.loader.poolSize:" +
        DEFAULT_MODEL_LOADER_POOL_SIZE +
        "}"
    ) int modelLoaderPoolSize,
    @Value(
      "${inference.models.loader.maxExecuteTime:" +
        DEFAULT_MODEL_LOADER_MAX_EXECUTE_TIME_MS +
        "}"
    ) long modelLoaderMaxExecuteTimeMs
  ) {
    this.vertx = vertx;
    this.modelPath = modelPath;
    this.modelLoaderPoolSize = modelLoaderPoolSize;
    this.modelLoaderMaxExecuteTimeMs = modelLoaderMaxExecuteTimeMs;
  }

  @Override
  protected String name() {
    return INFERENCE_SERVICE;
  }

  @Override
  protected void doStart() throws Exception {
    LOGGER.debug("Starting Inference service");
    super.doStart();
    // Loading a model takes seconds to minutes. A dedicated pool keeps it off the default worker
    // pool, and its max execute time keeps Vert.x from reporting a long load as a blocked thread.
    modelLoader = vertx.createSharedWorkerExecutor(
      MODEL_LOADER_POOL_NAME,
      modelLoaderPoolSize,
      modelLoaderMaxExecuteTimeMs,
      TimeUnit.MILLISECONDS
    );
    crudHandler = new ModelHandler(
      vertx,
      new HandlerRepository(),
      new ModelProviderRegistry(
        vertx,
        modelPath,
        modelLoaderScheduler(modelLoader)
      )
    );
    consumer = vertx
      .eventBus()
      .<Buffer>consumer(SERVICE_INFERENCE_MODELS_ADDRESS)
      .toObservable()
      .subscribe(crudHandler::handle, throwable ->
        LOGGER.error("Inference service handler failed", throwable)
      );
  }

  /**
   * Not ordered: an ordered scheduler, such as {@code RxHelper.blockingScheduler}, runs the tasks
   * of a context one after the other, so every model of the gateway would load in series.
   */
  public static Scheduler modelLoaderScheduler(WorkerExecutor modelLoader) {
    return new ContextScheduler(modelLoader.getDelegate(), false);
  }

  @Override
  protected void doStop() throws Exception {
    LOGGER.debug("Stopping Inference service");
    super.doStop();
    crudHandler.close();
    consumer.dispose();
    modelLoader
      .rxClose()
      .subscribe(
        () -> LOGGER.debug("Model loader pool closed"),
        throwable -> LOGGER.warn("Failed to close model loader pool", throwable)
      );
  }
}

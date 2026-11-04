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
package io.gravitee.inference.service.provider;

import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.InferenceService;
import io.gravitee.inference.service.model.LocalModelFactory;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.gravitee.reactive.webclient.api.ModelFetcher;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.core.Single;
import io.vertx.junit5.VertxExtension;
import io.vertx.junit5.VertxTestContext;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.WorkerExecutor;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

@ExtendWith(VertxExtension.class)
class HuggingFaceProviderTest {

  private static final String MODEL_NAME = "modelName";

  @TempDir
  Path modelPath;

  private Vertx vertx;
  private WorkerExecutor modelLoader;
  private LocalModelFactory modelFactory;
  private CyclicBarrier bothLoading;

  @BeforeEach
  void setUp(io.vertx.core.Vertx coreVertx) throws Exception {
    vertx = Vertx.newInstance(coreVertx);
    modelLoader = vertx.createSharedWorkerExecutor(
      "test-model-loader",
      2,
      60_000,
      MILLISECONDS
    );
    // Each load waits until the other one is loading too: it only completes if both run at once.
    bothLoading = new CyclicBarrier(2);
    modelFactory = mock(LocalModelFactory.class);
    when(modelFactory.build(any())).thenAnswer(invocation -> {
      bothLoading.await(2, SECONDS);
      return null;
    });
  }

  @AfterEach
  void tearDown() {
    modelLoader.rxClose().blockingAwait();
  }

  @Test
  void should_load_models_concurrently(VertxTestContext context) {
    var provider = provider(InferenceService.modelLoaderScheduler(modelLoader));
    var repository = new HandlerRepository();
    var loaded = context.checkpoint(2);

    // Both requests come from the same context, like START messages from the event bus consumer.
    vertx.runOnContext(v -> {
      provider
        .provide(startRequest("model-a"), repository)
        .subscribe(handler -> loaded.flag(), context::failNow);
      provider
        .provide(startRequest("model-b"), repository)
        .subscribe(handler -> loaded.flag(), context::failNow);
    });
  }

  @Test
  void ordered_blocking_scheduler_loads_models_in_series(
    VertxTestContext context
  ) {
    var provider = provider(RxHelper.blockingScheduler(vertx));
    var repository = new HandlerRepository();

    // The second load waits for the first, which times out waiting for the second.
    vertx.runOnContext(v -> {
      provider
        .provide(startRequest("model-a"), repository)
        .subscribe(
          handler -> context.failNow("the first load should time out"),
          error ->
            context.verify(() -> {
              assertThat(error).isInstanceOf(TimeoutException.class);
              context.completeNow();
            })
        );
      provider
        .provide(startRequest("model-b"), repository)
        .subscribe(handler -> {}, error -> {});
    });
  }

  private HuggingFaceProvider provider(Scheduler scheduler) {
    ModelFetcher fetcher = config ->
      Single.just(
        Map.of(
          ModelFileType.MODEL,
          "model.onnx",
          ModelFileType.TOKENIZER,
          "tokenizer.json"
        )
      );
    return new HuggingFaceProvider(
      modelPath.toString(),
      scheduler,
      fetcher,
      modelFactory
    );
  }

  private static InferenceRequest startRequest(String modelName) {
    return new InferenceRequest(
      InferenceAction.START,
      // The model name is removed from the payload that identifies the model: a second key keeps
      // the two models apart.
      Map.of(MODEL_NAME, modelName, "id", modelName)
    );
  }
}

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

import io.gravitee.inference.service.model.VllmModelFactory;
import io.gravitee.inference.vllm.VllmConfig;
import io.vertx.rxjava3.core.Vertx;
import java.util.Objects;

public final class VllmInferenceHandlerFactory
  implements InferenceHandlerFactory<VllmConfig> {

  private final Vertx vertx;
  private final VllmModelFactory modelFactory;

  public VllmInferenceHandlerFactory(
    Vertx vertx,
    VllmModelFactory modelFactory
  ) {
    this.vertx = Objects.requireNonNull(vertx, "vertx is required");
    this.modelFactory = Objects.requireNonNull(
      modelFactory,
      "modelFactory is required"
    );
  }

  @Override
  public InferenceHandler create(VllmConfig config) {
    int key = Objects.hash(
      config.model(),
      config.dtype(),
      config.maxModelLen(),
      config.maxNumSeqs(),
      config.quantization(),
      config.enableLora()
    );
    return new VllmInferenceHandler(vertx, config, modelFactory, key);
  }
}

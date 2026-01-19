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

import io.gravitee.inference.service.model.OnnxModelFactory;
import java.util.Map;
import java.util.Objects;

public final class LocalInferenceHandlerFactory implements InferenceHandlerFactory<Map<String, Object>> {

  private final OnnxModelFactory modelFactory;

  public LocalInferenceHandlerFactory(OnnxModelFactory modelFactory) {
    this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory is required");
  }

  @Override
  public InferenceHandler create(Map<String, Object> config) {
    return new LocalInferenceHandler(config, modelFactory);
  }
}

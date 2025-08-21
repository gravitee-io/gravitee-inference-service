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

import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.handler.InferenceHandler;
import io.vertx.rxjava3.core.Vertx;
import java.util.EnumMap;
import java.util.Map;
import java.util.Optional;

public class ModelProviderRegistry {

  private final Map<InferenceFormat, ModelProvider> providers = new EnumMap<>(InferenceFormat.class);
  private final Vertx vertx;
  private final String modelPath;

  public ModelProviderRegistry(Vertx vertx, String modelPath) {
    this.vertx = vertx;
    this.modelPath = modelPath;
    initializeProviders();
  }

  private void initializeProviders() {
    providers.put(InferenceFormat.ONNX_BERT, new HuggingFaceProvider(vertx, modelPath));
    // providers.put(InferenceFormat.REST_HTTP, new RestHttpProvider());
    providers.put(InferenceFormat.OPENAI, new OpenAIProvider());
  }

  public ModelProvider getProvider(InferenceFormat format) {
    return Optional
      .ofNullable(providers.get(format))
      .orElseThrow(() -> new IllegalArgumentException("No provider available for format: " + format));
  }
}

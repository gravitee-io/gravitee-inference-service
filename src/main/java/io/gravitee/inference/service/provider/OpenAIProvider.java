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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.rest.openai.embedding.EncodingFormat;
import io.gravitee.inference.rest.openai.embedding.OpenAIEmbeddingConfig;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.json.Json;
import java.net.URI;
import java.util.Map;

public class OpenAIProvider implements ModelProvider {

  record EmbeddingConfig(
    URI uri,
    String apiKey,
    String organizationId,
    String projectId,
    String model,
    Integer dimensions,
    EncodingFormat encodingFormat
  ) {}

  OpenAIProvider() {}

  @Override
  public Single<Model> loadModel(InferenceRequest inferenceRequest, ModelRepository repository) {
    return Single
      .just(inferenceRequest)
      .map(this::getOpenAIConfig)
      .map(finalConfig -> new ObjectMapper().convertValue(finalConfig, Map.class))
      .map(repository::add);
  }

  @Override
  public boolean isModelSupported() {
    return false;
  }

  OpenAIEmbeddingConfig getOpenAIConfig(InferenceRequest inferenceRequest) {
    EmbeddingConfig payload = Json.decodeValue(inferenceRequest.payload().toString(), EmbeddingConfig.class);
    return new OpenAIEmbeddingConfig(
      payload.uri,
      payload.apiKey,
      payload.organizationId,
      payload.projectId,
      payload.model,
      payload.dimensions,
      payload.encodingFormat
    );
  }
}

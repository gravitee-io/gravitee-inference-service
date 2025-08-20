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

import static io.gravitee.inference.api.Constants.INFERENCE_FORMAT;
import static io.gravitee.inference.api.Constants.INFERENCE_TYPE;

import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.rest.openai.embedding.EncodingFormat;
import io.gravitee.inference.rest.openai.embedding.OpenAIEmbeddingConfig;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.json.Json;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenAIProvider implements ModelProvider {

  private static final Logger LOGGER = LoggerFactory.getLogger(OpenAIProvider.class);

  record EmbeddingConfig(
    URI uri,
    String apiKey,
    String organizationId,
    String projectId,
    String model,
    Integer dimensions,
    EncodingFormat encodingFormat
  ) {
    static EmbeddingConfig fromInferenceRequest(InferenceRequest inferenceRequest) {
      Map<String, Object> payload = inferenceRequest.payload();

      return new EmbeddingConfig(
        URI.create((String) payload.get("uri")),
        (String) payload.get("apiKey"),
        (String) payload.get("organizationId"),
        (String) payload.get("projectId"),
        (String) payload.get("model"),
        payload.get("dimensions") != null ? ((Number) payload.get("dimensions")).intValue() : null,
        payload.get("encodingFormat") != null ? EncodingFormat.valueOf((String) payload.get("encodingFormat")) : null
      );
    }

    Map<String, Object> toMap() {
      Map<String, Object> map = new HashMap<>();

      map.put("uri", uri);
      map.put("apiKey", apiKey);
      map.put("model", model);

      if (organizationId != null) map.put("organizationId", organizationId);
      if (projectId != null) map.put("projectId", projectId);
      if (dimensions != null) map.put("dimensions", dimensions);
      if (encodingFormat != null) map.put("encodingFormat", encodingFormat.name());

      return map;
    }
  }

  OpenAIProvider() {}

  @Override
  public Single<Model<?>> loadModel(InferenceRequest inferenceRequest, ModelRepository repository) {
    return Single
      .just(inferenceRequest)
      .map(EmbeddingConfig::fromInferenceRequest)
      .map(EmbeddingConfig::toMap)
      .map(this::addInferenceInfo)
      .map(repository::add);
  }

  public Map<String, Object> addInferenceInfo(Map<String, Object> inferenceRequest) {
    inferenceRequest.put(INFERENCE_TYPE, InferenceType.EMBEDDING.name());
    inferenceRequest.put(INFERENCE_FORMAT, InferenceFormat.OPENAI.name());

    return inferenceRequest;
  }
}

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
import io.gravitee.inference.rest.http.embedding.HttpEmbeddingConfig;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
import io.vertx.rxjava3.core.Vertx;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HttpProvider implements ModelProvider {

  private static final Logger LOGGER = LoggerFactory.getLogger(HttpProvider.class);
  Vertx vertx;

  @Override
  public Single<Model<?>> loadModel(InferenceRequest inferenceRequest, ModelRepository repository) {
    return Single
      .just(inferenceRequest.payload())
      .map(payload ->
        new HttpEmbeddingConfig(
          URI.create((String) payload.get("uri")),
          HttpMethod.valueOf((String) payload.get("method")),
          parseHeaders(payload, ("headers")),
          (String) payload.get("requestBodyTemplate"),
          (String) payload.get("inputLocation"),
          (String) payload.get("outputEmbeddingLocation")
        )
      )
      .map(this::httpEmbeddingConfigToMap)
      .map(this::addInferenceInfo)
      .map(repository::add);
  }

  private static Map<String, String> parseHeaders(Map<String, Object> payload, String key) {
    Object value = payload.get(key);
    return value instanceof Map<?, ?> map ? (Map<String, String>) map : Map.of();
  }

  Map<String, Object> httpEmbeddingConfigToMap(HttpEmbeddingConfig config) {
    Map<String, Object> map = new HashMap<>();
    map.put("uri", config.getUri());
    map.put("method", config.getMethod());
    map.put("headers", config.getHeaders());
    map.put("requestBodyTemplate", config.getRequestBodyTemplate());
    map.put("inputLocation", config.getInputLocation());
    map.put("outputEmbeddingLocation", config.getOutputEmbeddingLocation());
    return map;
  }

  public Map<String, Object> addInferenceInfo(Map<String, Object> inferenceRequest) {
    inferenceRequest.put(INFERENCE_TYPE, InferenceType.EMBEDDING.name());
    inferenceRequest.put(INFERENCE_FORMAT, InferenceFormat.HTTP.name());

    return inferenceRequest;
  }
}

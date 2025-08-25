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
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.RemoteInferenceHandler;
import io.gravitee.inference.service.model.RemoteModelFactory;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
import io.vertx.rxjava3.core.Vertx;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HttpProvider implements InferenceHandlerProvider {

  private static final Logger LOGGER = LoggerFactory.getLogger(HttpProvider.class);
  private final RemoteModelFactory modelFactory;

  HttpProvider(Vertx vertx) {
    this.modelFactory = new RemoteModelFactory(vertx);
  }

  private static Map<String, String> parseHeaders(Map<String, Object> payload, String key) {
    Object value = payload.get(key);
    return value instanceof Map<?, ?> map ? (Map<String, String>) map : Map.of();
  }

  Map<String, Object> requestToConfigToMap(InferenceRequest request) {
    Map<String, Object> payload = request.payload();

    HttpEmbeddingConfig config = new HttpEmbeddingConfig(
      URI.create((String) payload.get("uri")),
      HttpMethod.valueOf((String) payload.get("method")),
      parseHeaders(payload, ("headers")),
      (String) payload.get("requestBodyTemplate"),
      (String) payload.get("inputLocation"),
      (String) payload.get("outputEmbeddingLocation")
    );

    Map<String, Object> map = new HashMap<>();
    map.put("uri", config.getUri());
    map.put("method", config.getMethod());
    map.put("headers", config.getHeaders());
    map.put("requestBodyTemplate", config.getRequestBodyTemplate());
    map.put("inputLocation", config.getInputLocation());
    map.put("outputEmbeddingLocation", config.getOutputEmbeddingLocation());

    return map;
  }

  private InferenceHandler getInferenceHandler(Map<String, Object> inferenceRequest, HandlerRepository repository) {
    inferenceRequest.put(INFERENCE_TYPE, InferenceType.EMBEDDING.name());
    inferenceRequest.put(INFERENCE_FORMAT, InferenceFormat.HTTP.name());

    return repository.add(new RemoteInferenceHandler(inferenceRequest, modelFactory));
  }

  @Override
  public Single<InferenceHandler> provide(InferenceRequest inferenceRequest, HandlerRepository repository) {
    return Single
      .just(inferenceRequest)
      .map(this::requestToConfigToMap)
      .map(map -> this.getInferenceHandler(map, repository));
  }
}

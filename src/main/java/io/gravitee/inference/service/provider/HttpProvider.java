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
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.rest.http.embedding.HttpEmbeddingConfig;
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.InferenceHandlerFactory;
import io.gravitee.inference.service.handler.RemoteInferenceHandlerFactory;
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

  static final String URI = "uri";
  static final String METHOD = "method";
  static final String HEADERS = "headers";
  static final String REQUEST_BODY_TEMPLATE = "requestBodyTemplate";
  static final String INPUT_LOCATION = "inputLocation";
  static final String OUTPUT_EMBEDDING_LOCATION = "outputEmbeddingLocation";

  private final RemoteInferenceHandlerFactory handlerFactory;

  HttpProvider(Vertx vertx) {
    this.handlerFactory = new RemoteInferenceHandlerFactory(new RemoteModelFactory(vertx));
  }

  @Override
  public Single<InferenceHandler> provide(InferenceRequest inferenceRequest, HandlerRepository repository) {
    return Single
      .just(inferenceRequest)
      .map(this::requestToConfigToMap)
      .map(map -> repository.add(handlerFactory.create(map)));
  }

  @Override
  public InferenceHandlerFactory<?> factory() {
    return handlerFactory;
  }

  Map<String, Object> requestToConfigToMap(InferenceRequest request) {
    Map<String, Object> payload = request.payload();
    var wrapper = new ConfigWrapper(payload);

    var config = new HttpEmbeddingConfig(
      java.net.URI.create(wrapper.get(URI)),
      HttpMethod.valueOf(wrapper.get(METHOD)),
      parseHeaders(payload, HEADERS),
      wrapper.get(REQUEST_BODY_TEMPLATE),
      wrapper.get(INPUT_LOCATION),
      wrapper.get(OUTPUT_EMBEDDING_LOCATION)
    );

    return Map.of(
      URI,
      config.getUri(),
      METHOD,
      config.getMethod(),
      HEADERS,
      config.getHeaders(),
      REQUEST_BODY_TEMPLATE,
      config.getRequestBodyTemplate(),
      INPUT_LOCATION,
      config.getInputLocation(),
      OUTPUT_EMBEDDING_LOCATION,
      config.getOutputEmbeddingLocation(),
      INFERENCE_FORMAT,
      wrapper.get(INFERENCE_FORMAT),
      INFERENCE_TYPE,
      wrapper.get(INFERENCE_TYPE)
    );
  }

  private static Map<String, String> parseHeaders(Map<String, Object> payload, String key) {
    Object value = payload.get(key);
    return value instanceof Map<?, ?> map ? (Map<String, String>) map : Map.of();
  }
}

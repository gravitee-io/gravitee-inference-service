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
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.RemoteInferenceHandler;
import io.gravitee.inference.service.model.RemoteModelFactory;
import io.gravitee.inference.service.provider.config.EmbeddingConfig;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;

public class OpenAIProvider implements InferenceHandlerProvider {

  private final RemoteModelFactory modelFactory;

  public OpenAIProvider(Vertx vertx) {
    this.modelFactory = new RemoteModelFactory(vertx);
  }

  @Override
  public Single<InferenceHandler> provide(InferenceRequest inferenceRequest, HandlerRepository repository) {
    return Single
      .just(inferenceRequest)
      .map(request -> {
        var config = EmbeddingConfig.fromInferenceRequest(request).toMap();
        config.put(INFERENCE_TYPE, request.payload().get(INFERENCE_TYPE));
        config.put(INFERENCE_FORMAT, request.payload().get(INFERENCE_FORMAT));
        return config;
      })
      .map(map -> repository.add(new RemoteInferenceHandler(map, modelFactory)));
  }
}

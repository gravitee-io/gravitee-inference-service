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
package io.gravitee.inference.service;

import static io.gravitee.inference.api.Constants.INFERENCE_FORMAT;
import static io.gravitee.inference.api.Constants.INFERENCE_TYPE;
import static io.gravitee.inference.api.Constants.SERVICE_INFERENCE_MODELS_ADDRESS;

import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.rest.openai.embedding.EncodingFormat;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ServiceOpenaiEmbeddingTest extends ServiceEmbeddingTest {

  protected static final Logger LOGGER = LoggerFactory.getLogger(ServiceOpenaiEmbeddingTest.class);

  private static final String SERVICE_URL = "http://localhost:11434/v1";
  private static final String MODEL_NAME = "all-minilm:latest";
  public static final String URI = "uri";
  public static final String API_KEY = "apiKey";
  public static final String MODEL = "model";
  public static final String ENCODING_FORMAT = "encodingFormat";

  @Override
  String loadModel() {
    InferenceRequest openaiStartRequest = new InferenceRequest(
      InferenceAction.START,
      Map.of(
        INFERENCE_FORMAT,
        "OPENAI",
        INFERENCE_TYPE,
        "EMBEDDING",
        URI,
        java.net.URI.create(SERVICE_URL),
        API_KEY,
        "FAKE_KEY",
        MODEL,
        MODEL_NAME,
        ENCODING_FORMAT,
        EncodingFormat.FLOAT.name()
      )
    );

    return vertx
      .eventBus()
      .<Object>request(SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(openaiStartRequest))
      .blockingGet()
      .body()
      .toString();
  }
}

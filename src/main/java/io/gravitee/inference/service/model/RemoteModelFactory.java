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
package io.gravitee.inference.service.model;

import static io.gravitee.inference.api.Constants.INFERENCE_FORMAT;
import static io.gravitee.inference.api.Constants.INFERENCE_TYPE;

import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.rest.RestInference;
import io.gravitee.inference.rest.http.embedding.HttpEmbeddingConfig;
import io.gravitee.inference.rest.http.embedding.HttpEmbeddingInference;
import io.gravitee.inference.rest.openai.embedding.OpenAIEmbeddingConfig;
import io.gravitee.inference.rest.openai.embedding.OpenaiEmbeddingInference;
import io.vertx.rxjava3.core.Vertx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class RemoteModelFactory implements InferenceModelFactory<RestInference<?, ?, ?>> {

  private static final Logger LOGGER = LoggerFactory.getLogger(RemoteModelFactory.class);

  public static final String URI = "uri";
  public static final String OPENAI_API_KEY = "apiKey";
  public static final String OPENAI_DIMENSIONS = "dimensions";
  public static final String OPENAI_MODEL = "model";
  public static final String OPENAI_PROJECT_ID = "projectId";
  public static final String OPENAI_ORGANIZATION_ID = "organizationId";
  public static final String HTTP_METHOD = "method";
  public static final String HTTP_HEADERS = "headers";
  public static final String HTTP_REQUEST_BODY_TEMPLATE = "requestBodyTemplate";
  public static final String HTTP_INPUT_LOCATION = "inputLocation";
  public static final String HTTP_OUTPUT_EMBEDDING_LOCATION = "outputEmbeddingLocation";

  private final Vertx vertx;

  public RemoteModelFactory(Vertx vertx) {
    this.vertx = vertx;
  }

  public RestInference<?, ?, ?> build(ConfigWrapper config) {
    InferenceType type = InferenceType.valueOf(config.get(INFERENCE_TYPE));
    InferenceFormat format = InferenceFormat.valueOf(config.get(INFERENCE_FORMAT));
    return switch (type) {
      case EMBEDDING -> switch (format) {
        case OPENAI -> createOpenAIEmbeddingInference(config);
        case HTTP -> createHttpEmbeddingInference(config);
        default -> throw new IllegalArgumentException(
          String.format(
            "Unsupported inference format '%s' for type EMBEDDING. Supported formats: [ONNX_BERT, OPENAI]",
            format
          )
        );
      };
      default -> throw new IllegalArgumentException(String.format("Unsupported inference type '%s'", type));
    };
  }

  private HttpEmbeddingInference createHttpEmbeddingInference(ConfigWrapper config) {
    HttpEmbeddingConfig httpEmbeddingConfig = new HttpEmbeddingConfig(
      config.get(URI),
      config.get(HTTP_METHOD),
      config.get(HTTP_HEADERS),
      config.get(HTTP_REQUEST_BODY_TEMPLATE),
      config.get(HTTP_INPUT_LOCATION),
      config.get(HTTP_OUTPUT_EMBEDDING_LOCATION)
    );
    var inferenceService = new HttpEmbeddingInference(httpEmbeddingConfig, vertx);
    LOGGER.debug("Http Embedding inference service started {} with config {}", inferenceService, httpEmbeddingConfig);
    return inferenceService;
  }

  private OpenaiEmbeddingInference createOpenAIEmbeddingInference(ConfigWrapper config) {
    OpenAIEmbeddingConfig openAIEmbeddingConfig = new OpenAIEmbeddingConfig(
      config.get(URI),
      config.get(OPENAI_API_KEY),
      config.get(OPENAI_ORGANIZATION_ID),
      config.get(OPENAI_PROJECT_ID),
      config.get(OPENAI_MODEL),
      config.get(OPENAI_DIMENSIONS)
    );
    var inferenceService = new OpenaiEmbeddingInference(openAIEmbeddingConfig, vertx);
    LOGGER.debug("OpenAI Embedding inference service started {} with config {}", inferenceService, openAIEmbeddingConfig);
    return inferenceService;
  }
}

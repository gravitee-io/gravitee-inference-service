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
package io.gravitee.inference.service.repository;

import static io.gravitee.inference.api.Constants.*;
import static io.gravitee.inference.api.service.InferenceAction.START;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.embedding.OnnxBertEmbeddingModel;
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.LocalInferenceHandler;
import io.gravitee.inference.service.model.InferenceModelFactory;
import io.gravitee.inference.service.model.LocalModelFactory;
import io.vertx.rxjava3.core.Vertx;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ModelRepositoryTest {

  private static final String SEQUENCE_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english";
  private static final String TOKEN_MODEL = "dslim/distilbert-NER";
  public static final String ONNX_MODEL = "/resolve/main/onnx/model.onnx";
  public static final String TOKENIZER_JSON = "/resolve/main/onnx/tokenizer.json";
  public static final String CONFIG_JSON = "/resolve/main/config.json";

  private HandlerRepository repository;

  public static Stream<Arguments> params_that_must_build_model() {
    return Stream.of(
      Arguments.of(
        new InferenceRequest(
          START,
          Map.of(
            INFERENCE_FORMAT,
            "ONNX_BERT",
            INFERENCE_TYPE,
            "CLASSIFIER",
            MODEL_PATH,
            "model.onnx",
            TOKENIZER_PATH,
            "tokenizer.json",
            CONFIG_JSON_PATH,
            "config.json",
            CLASSIFIER_MODE,
            "TOKEN"
          )
        )
      ),
      Arguments.of(
        new InferenceRequest(
          START,
          Map.of(
            INFERENCE_FORMAT,
            "ONNX_BERT",
            INFERENCE_TYPE,
            "CLASSIFIER",
            MODEL_PATH,
            "model.onnx",
            TOKENIZER_PATH,
            "tokenizer.json",
            CONFIG_JSON_PATH,
            "config.json",
            CLASSIFIER_MODE,
            "SEQUENCE"
          )
        )
      ),
      Arguments.of(
        new InferenceRequest(
          START,
          Map.of(
            INFERENCE_FORMAT,
            "ONNX_BERT",
            INFERENCE_TYPE,
            "EMBEDDING",
            MODEL_PATH,
            "model.quantized.onnx",
            TOKENIZER_PATH,
            "tokenizer.json",
            POOLING_MODE,
            "MEAN",
            MAX_SEQUENCE_LENGTH,
            MAX_SEQUENCE_LENGTH_DEFAULT_VALUE
          )
        )
      )
    );
  }

  @BeforeEach
  void setUp() {
    repository = new HandlerRepository();
  }

  @ParameterizedTest
  @MethodSource("params_that_must_build_model")
  void must_setup_model_lifecycle(InferenceRequest request) {
    LocalInferenceHandler handler = new LocalInferenceHandler(request.payload(), mock(LocalModelFactory.class));
    InferenceHandler model = repository.add(handler);
    assertNotNull(model);

    assertEquals(1, repository.getModelsSize());
    assertEquals(1, repository.getModelUsage(model.key()));

    repository.add(handler);

    assertEquals(1, repository.getModelsSize());
    assertEquals(2, repository.getModelUsage(model.key()));

    repository.remove(handler);

    assertEquals(1, repository.getModelsSize());
    assertEquals(1, repository.getModelUsage(model.key()));

    repository.remove(model);

    assertEquals(0, repository.getModelsSize());
    assertEquals(0, repository.getModelUsage(model.key()));
  }
}

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
package io.gravitee.inference.service.handler.action;

import static io.gravitee.inference.api.Constants.*;
import static io.gravitee.inference.api.service.InferenceAction.CREATE;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.embedding.OnnxBertEmbeddingModel;
import java.net.MalformedURLException;
import java.util.List;
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
public class CreateModelActionTest extends BaseDownloadModelTest {

  private static final String SEQUENCE_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english";
  private static final String TOKEN_MODEL = "dslim/distilbert-NER";
  public static final String ONNX_MODEL = "/resolve/main/onnx/model.onnx";
  public static final String TOKENIZER_JSON = "/resolve/main/onnx/tokenizer.json";

  private CreateModelAction actionHandler;

  public static Stream<Arguments> params_that_must_build_model() throws MalformedURLException {
    return Stream.of(
      Arguments.of(
        new InferenceRequest(
          CREATE,
          Map.of(
            INFERENCE_FORMAT,
            "ONNX_BERT",
            INFERENCE_TYPE,
            "CLASSIFIER",
            MODEL_PATH,
            getUriIfExist(TOKEN_MODEL, ONNX_MODEL).toASCIIString().split(":")[1],
            TOKENIZER_PATH,
            getUriIfExist(TOKEN_MODEL, TOKENIZER_JSON).toASCIIString().split(":")[1],
            CLASSIFIER_MODE,
            "TOKEN",
            CLASSIFIER_LABELS,
            List.of("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC")
          )
        ),
        "My name is Clara and I am from Berkley, California",
        OnnxBertClassifierModel.class
      ),
      Arguments.of(
        new InferenceRequest(
          CREATE,
          Map.of(
            INFERENCE_FORMAT,
            "ONNX_BERT",
            INFERENCE_TYPE,
            "CLASSIFIER",
            MODEL_PATH,
            getUriIfExist(SEQUENCE_MODEL, ONNX_MODEL).toASCIIString().split(":")[1],
            TOKENIZER_PATH,
            getUriIfExist(SEQUENCE_MODEL, TOKENIZER_JSON).toASCIIString().split(":")[1],
            CLASSIFIER_MODE,
            "SEQUENCE",
            CLASSIFIER_LABELS,
            List.of("Negative", "Positive")
          )
        ),
        "I am happy today!",
        OnnxBertClassifierModel.class
      ),
      Arguments.of(
        new InferenceRequest(
          CREATE,
          Map.of(
            INFERENCE_FORMAT,
            "ONNX_BERT",
            INFERENCE_TYPE,
            "EMBEDDING",
            MODEL_PATH,
            getUriIfExist("Xenova/all-MiniLM-L6-v2", "/resolve/main/onnx/model_quantized.onnx")
              .toASCIIString()
              .split(":")[1],
            TOKENIZER_PATH,
            getUriIfExist("Xenova/all-MiniLM-L6-v2", "/resolve/main/tokenizer.json").toASCIIString().split(":")[1],
            POOLING_MODE,
            "MEAN",
            MAX_SEQUENCE_LENGTH,
            MAX_SEQUENCE_LENGTH_DEFAULT_VALUE
          )
        ),
        "The big brown fox jumps over the lazy dog",
        OnnxBertEmbeddingModel.class
      )
    );
  }

  @BeforeEach
  void setUp() {
    actionHandler = new CreateModelAction();
  }

  @ParameterizedTest
  @MethodSource("params_that_must_build_model")
  void must_build_model(InferenceRequest request, String input, Class<?> expectedClass) {
    InferenceModel<?, String, ?> model = actionHandler.handle(request);
    assertNotNull(model);

    assertInstanceOf(expectedClass, model);
    assertNotNull(model.infer(input));
  }
}

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
package io.gravitee.inference.service.integration.embedding;

import static io.gravitee.inference.api.Constants.CONFIG_JSON_PATH;
import static io.gravitee.inference.api.Constants.INFERENCE_FORMAT;
import static io.gravitee.inference.api.Constants.INFERENCE_TYPE;
import static io.gravitee.inference.api.Constants.MAX_SEQUENCE_LENGTH;
import static io.gravitee.inference.api.Constants.MAX_SEQUENCE_LENGTH_DEFAULT_VALUE;
import static io.gravitee.inference.api.Constants.MODEL_PATH;
import static io.gravitee.inference.api.Constants.POOLING_MODE;
import static io.gravitee.inference.api.Constants.SERVICE_INFERENCE_MODELS_ADDRESS;
import static io.gravitee.inference.api.Constants.TOKENIZER_PATH;

import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.vertx.core.json.Json;
import java.util.Map;

public class ServiceBertEmbeddingTest extends ServiceEmbeddingTest {

  @Override
  String loadModel() {
    InferenceRequest localStartRequest = new InferenceRequest(
      InferenceAction.START,
      Map.of(
        INFERENCE_FORMAT,
        "ONNX_BERT",
        INFERENCE_TYPE,
        "EMBEDDING",
        "modelName",
        "Xenova/bge-small-en-v1.5",
        MODEL_PATH,
        "onnx/model_quantized.onnx",
        TOKENIZER_PATH,
        "tokenizer.json",
        CONFIG_JSON_PATH,
        "config.json",
        POOLING_MODE,
        "MEAN",
        MAX_SEQUENCE_LENGTH,
        MAX_SEQUENCE_LENGTH_DEFAULT_VALUE
      )
    );

    return vertx
      .eventBus()
      .<Object>request(
        SERVICE_INFERENCE_MODELS_ADDRESS,
        Json.encodeToBuffer(localStartRequest)
      )
      .blockingGet()
      .body()
      .toString();
  }

  @Override
  Integer waitTime() {
    return 20_000;
  }
}

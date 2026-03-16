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
package io.gravitee.inference.service.handler;

import io.gravitee.inference.api.textgen.AbstractBatchEngine;
import io.gravitee.inference.llama.cpp.EventBusUtils;
import io.gravitee.inference.llama.cpp.ModelConfig;
import io.gravitee.inference.llama.cpp.Request;
import io.gravitee.inference.llama.cpp.TagConfig;
import io.gravitee.inference.service.model.LlamaCppModelFactory;
import io.vertx.rxjava3.core.Vertx;
import java.util.Map;

public class LlamaCppInferenceHandler
  extends AbstractStreamingInferenceHandler<Request> {

  private final ModelConfig modelConfig;
  private final LlamaCppModelFactory modelFactory;

  public LlamaCppInferenceHandler(
    Vertx vertx,
    ModelConfig modelConfig,
    LlamaCppModelFactory modelFactory,
    int key
  ) {
    super(vertx, key, "llama-cpp-stream-cleanup");
    this.modelConfig = modelConfig;
    this.modelFactory = modelFactory;
  }

  @Override
  protected AbstractBatchEngine<?, Request, String, ?> buildEngine() {
    return modelFactory.build(modelConfig, this::publishToken);
  }

  @Override
  @SuppressWarnings("unchecked")
  protected Request buildRequest(Map<String, Object> payload) {
    var baseRequest = new Request(payload);

    TagConfig reasoningTags = payload.get("reasoningTags") != null
      ? parseTagConfig(
        (Map<String, Object>) payload.get("reasoningTags"),
        TagConfig::new
      )
      : null;
    TagConfig toolTags = payload.get("toolTags") != null
      ? parseTagConfig(
        (Map<String, Object>) payload.get("toolTags"),
        TagConfig::new
      )
      : null;

    return new Request(
      baseRequest.prompt(),
      baseRequest.messages(),
      baseRequest.maxTokens(),
      baseRequest.temperature(),
      baseRequest.topP(),
      baseRequest.presencePenalty(),
      baseRequest.frequencyPenalty(),
      baseRequest.stop(),
      baseRequest.seed(),
      reasoningTags,
      toolTags
    );
  }

  @Override
  protected String tokensAddress(String streamId, int seqId) {
    return EventBusUtils.tokensAddress(streamId, seqId);
  }
}

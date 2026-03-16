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
import io.gravitee.inference.service.model.VllmModelFactory;
import io.gravitee.inference.vllm.EventBusUtils;
import io.gravitee.inference.vllm.TagConfig;
import io.gravitee.inference.vllm.VllmConfig;
import io.gravitee.inference.vllm.VllmRequest;
import io.vertx.rxjava3.core.Vertx;
import java.util.Map;

public class VllmInferenceHandler
  extends AbstractStreamingInferenceHandler<VllmRequest> {

  private final VllmConfig vllmConfig;
  private final VllmModelFactory modelFactory;

  public VllmInferenceHandler(
    Vertx vertx,
    VllmConfig vllmConfig,
    VllmModelFactory modelFactory,
    int key
  ) {
    super(vertx, key, "vllm-stream-cleanup");
    this.vllmConfig = vllmConfig;
    this.modelFactory = modelFactory;
  }

  @Override
  protected AbstractBatchEngine<?, VllmRequest, String, ?> buildEngine() {
    return modelFactory.build(vllmConfig, this::publishToken);
  }

  @Override
  @SuppressWarnings("unchecked")
  protected VllmRequest buildRequest(Map<String, Object> payload) {
    var baseRequest = new VllmRequest(payload);

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

    return new VllmRequest(
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
      toolTags,
      baseRequest.tools(),
      baseRequest.loraName(),
      baseRequest.loraPath()
    );
  }

  @Override
  protected String tokensAddress(String streamId, int seqId) {
    return EventBusUtils.tokensAddress(streamId, seqId);
  }
}

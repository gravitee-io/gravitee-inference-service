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
package io.gravitee.inference.service.provider.config;

import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.rest.openai.embedding.EncodingFormat;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

public record EmbeddingConfig(
  URI uri,
  String apiKey,
  String organizationId,
  String projectId,
  String model,
  Integer dimensions,
  EncodingFormat encodingFormat
) {
  static final String URI = "uri";
  static final String API_KEY = "apiKey";
  static final String ORGANIZATION_ID = "organizationId";
  static final String PROJECT_ID = "projectId";
  static final String MODEL = "model";
  static final String DIMENSIONS = "dimensions";
  static final String ENCODING_FORMAT = "encodingFormat";

  public static EmbeddingConfig fromInferenceRequest(
    InferenceRequest inferenceRequest
  ) {
    Map<String, Object> payload = inferenceRequest.payload();

    return new EmbeddingConfig(
      java.net.URI.create((String) payload.get(URI)),
      (String) payload.get(API_KEY),
      (String) payload.get(ORGANIZATION_ID),
      (String) payload.get(PROJECT_ID),
      (String) payload.get(MODEL),
      payload.get(DIMENSIONS) != null
        ? ((Number) payload.get(DIMENSIONS)).intValue()
        : null,
      payload.get(ENCODING_FORMAT) != null
        ? EncodingFormat.valueOf((String) payload.get(ENCODING_FORMAT))
        : null
    );
  }

  public Map<String, Object> toMap() {
    Map<String, Object> map = new HashMap<>();

    map.put(URI, uri);
    map.put(API_KEY, apiKey);
    map.put(MODEL, model);

    if (organizationId != null) map.put(ORGANIZATION_ID, organizationId);
    if (projectId != null) map.put(PROJECT_ID, projectId);
    if (dimensions != null) map.put(DIMENSIONS, dimensions);
    if (encodingFormat != null) map.put(ENCODING_FORMAT, encodingFormat.name());
    return map;
  }
}

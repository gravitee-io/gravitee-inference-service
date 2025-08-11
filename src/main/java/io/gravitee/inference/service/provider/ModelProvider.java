package io.gravitee.inference.service.provider;

import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
import io.reactivex.rxjava3.core.Single;

public interface ModelProvider {
  Single<Model> loadModel(InferenceRequest inferenceRequest, ModelRepository repository);

  boolean isModelSupported();
}

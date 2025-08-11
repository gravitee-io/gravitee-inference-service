package io.gravitee.inference.service.provider;

import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.vertx.rxjava3.core.Vertx;
import java.util.EnumMap;
import java.util.Map;
import java.util.Optional;

public class ModelProviderRegistry {

  private final Map<InferenceFormat, ModelProvider> providers = new EnumMap<>(InferenceFormat.class);
  private final Vertx vertx;
  private final String modelPath;

  public ModelProviderRegistry(Vertx vertx, String modelPath) {
    this.vertx = vertx;
    this.modelPath = modelPath;
    initializeProviders();
  }

  private void initializeProviders() {
    providers.put(InferenceFormat.ONNX_BERT, new HuggingFaceProvider(vertx, modelPath));
    // providers.put(InferenceFormat.REST_HTTP, new RestHttpProvider());
    // providers.put(InferenceFormat.REST_OPENAI, new RestOpenAIProvider());
  }

  public ModelProvider getProvider(InferenceRequest request) {
    InferenceFormat format = determineFormat(request);
    return Optional
      .ofNullable(providers.get(format))
      .orElseThrow(() -> new IllegalArgumentException("No provider available for format: " + format));
  }

  public ModelProvider getProvider(InferenceFormat format) {
    return Optional
      .ofNullable(providers.get(format))
      .orElseThrow(() -> new IllegalArgumentException("No provider available for format: " + format));
  }

  public void registerProvider(InferenceFormat format, ModelProvider provider) {
    providers.put(format, provider);
  }

  private InferenceFormat determineFormat(InferenceRequest request) {
    ConfigWrapper config = new ConfigWrapper(request.payload());
    String formatStr = String.valueOf(config.get("format", String.class));

    if (formatStr != null) {
      return InferenceFormat.valueOf(formatStr.toUpperCase());
    }
    throw new IllegalArgumentException("Invalid format: " + formatStr);
  }
}

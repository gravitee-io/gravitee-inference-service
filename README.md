# gravitee-inference-service

`gravitee-inference-service` is a plugin for the Gravitee platform that enables seamless integration of machine learning models via Vert.x. This service allows for model loading and querying through a reactive event-based architecture.

## Requirements

- Java 21
- Maven (`mvn`)
- Vert.x

## Features

- Load machine learning models using the event bus.
- Query models with a flexible and reactive programming style.
- Leverages Vert.x’s concurrency model for efficient, non-blocking requests.

## Installation

Ensure that the following dependencies are available in your environment:

- `gravitee-inference` (for model support)
- Java 21 and Maven (`mvn`)
- Vert.x (for event-driven architecture)

## Setup

To install and configure the plugin, you will need to integrate this service into your existing Gravitee setup. Ensure that Vert.x is running in your environment, as this service depends on it.

To built it:
```bash
$ mvn clean install
```

## Example Usage

The following example demonstrates how to interact with the `gravitee-inference-service` plugin via Vert.x:

### Java Code Example

#### 1. Load the Model

The first step is to load the model by sending a request to the `SERVICE_INFERENCE_MODELS_ADDRESS` address. The response will contain the address to query the model.

```java
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.RxHelper;
import io.vertx.core.json.Json;

Vertx vertx = Vertx.vertx();

vertx.eventBus().<Buffer>request(SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(request))
        // Model may take some time to load depending on the size and for the first time
        .subscribeOn(RxHelper.blockingScheduler(vertx.getDelegate()))
        .observeOn(RxHelper.scheduler(vertx.getDelegate()))
        .map(message -> message.body().toString()) // The response will be the model's address
        .subscribe(address -> {
            System.out.println("Model address: " + address); // Store or process the address
        }, Throwable::printStackTrace);
```

This request sends a model loading command and returns the model's address.

#### 2. Query the Model

Once the model is loaded, you can query it by sending a second request to the model’s address. The request will include the input data for the model.

```java
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.RxHelper;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.gravitee.inference.api.service.InferenceAction;

import static io.gravitee.inference.api.Constants.INPUT;

vertx
    .eventBus()
    .<Buffer>request(
        modelAddress, // The address returned from the first request
        Json.encodeToBuffer(new InferenceRequest(InferenceAction.INFER, Map.of(INPUT, "The big brown fox jumps over the lazy dog")))
    )
    .subscribeOn(RxHelper.blockingScheduler(vertx.getDelegate()))
    .observeOn(RxHelper.scheduler(vertx.getDelegate()))
    .map(message -> Json.decodeValue(message.body(), clazz)) // Decode the result
    .subscribe(
        result -> {
            System.out.println("Inference Result: " + result); // Process or display the result
        },
        Throwable::printStackTrace
    );
```

This call sends the input data (e.g., `"The big brown fox jumps over the lazy dog"`) to the model for inference.

---

### Flow Explanation

1. **Model Creation (First Call)**:
    - The first event bus request (`SERVICE_INFERENCE_MODELS_ADDRESS`) initializes the model loading process and returns an address.
2. **Model Query (Second Call)**:
    - The second request queries the model using the address obtained from the first call. It sends the input data (e.g., `"The big brown fox jumps over the lazy dog"`) to the model for inference.
3. **Result Handling**:
    - The model's output is returned and processed by decoding the message and outputting or further handling the result.

### Request Examples

#### Request for Embeddings

The request for embedding requires a model and tokenizer file. Below is how you can START the request with a dynamically generated random path:

```java
import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.embedding.PoolingMode;

import java.nio.file.Paths;
import java.util.Map;
import java.util.List;
import java.util.UUID;
import io.gravitee.inference.api.service.InferenceAction;

import static io.gravitee.inference.api.service.InferenceFormat.ONNX_BERT;
import static io.gravitee.inference.api.service.InferenceType.EMBEDDING;
import static io.gravitee.inference.api.classifier.ClassifierMode.TOKEN;
import static io.gravitee.inference.api.Constants.*;

var request = new InferenceRequest(
  InferenceAction.START,
  Map.of(
    INFERENCE_FORMAT, ONNX_BERT,
    INFERENCE_TYPE, EMBEDDING,
    MODEL_PATH, "/path/to/your/Xenova/all-MiniLM-L6-v2/model_quantized.onnx",
    TOKENIZER_PATH, "/path/to/your/Xenova/all-MiniLM-L6-v2/tokenizer.json",
    Constants.POOLING_MODE, PoolingMode.MEAN,
    MAX_SEQUENCE_LENGTH, 512
  )
);
```

#### Request for Classification

For the classification request, the process is similar but with different labels for token classification:

```java
import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceRequest;

import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import io.gravitee.inference.api.service.InferenceAction;

import static io.gravitee.inference.api.service.InferenceFormat.ONNX_BERT;
import static io.gravitee.inference.api.service.InferenceType.CLASSIFIER;
import static io.gravitee.inference.api.classifier.ClassifierMode.TOKEN;
import static io.gravitee.inference.api.Constants.*;

var request =
  new InferenceRequest(
    InferenceAction.START,
    Map.of(
      INFERENCE_FORMAT, ONNX_BERT,
      INFERENCE_TYPE, CLASSIFIER,
      CLASSIFIER_MODE, TOKEN,
      MODEL_PATH, "/path/to/your/dslim/distilbert-NER/model.onnx",
      TOKENIZER_PATH, "/path/to/your/dslim/distilbert-NER/tokenizer.json",
      CLASSIFIER_LABELS, List.of("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"),
      DISCARDED_LABELS, List.of("O")
    )
  );
```

> If you provide the same model at least twice (based on the configuration map), many addresses will be created but only
> one model will be loaded in memory

#### Teardown the inference

In order to teardown the inference, just build a "STOP" inference request and provide the inference model
address.


```java
import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceRequest;

import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import io.gravitee.inference.api.service.InferenceAction;

import static io.gravitee.inference.api.service.InferenceFormat.ONNX_BERT;
import static io.gravitee.inference.api.service.InferenceType.CLASSIFIER;
import static io.gravitee.inference.api.classifier.ClassifierMode.TOKEN;
import static io.gravitee.inference.api.Constants.*;

var request =
  new InferenceRequest(
    InferenceAction.STOP,
    Map.of(
      MODEL_ADDRESS_KEY, "<inference-model-address>"
    )
  );
```

> If you provided the same model several times, stopping the inference will just teardown the address
> but the model will be kept in memory until no address are bound to the model.
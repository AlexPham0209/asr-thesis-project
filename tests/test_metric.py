import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Assuming your function is saved in a file named core_metrics.py
from src.metrics.asr_metric import create_metric


@pytest.fixture
def mock_processor():
    """Fixtures to mock the Hugging Face processor and tokenizer."""
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    # Mock batch_decode to return predictable strings based on the input
    processor.batch_decode = MagicMock(
        side_effect=lambda ids, **kwargs: [f"decoded_string_{i}" for i in range(len(ids))]
    )
    return processor


@pytest.fixture
def mock_eval_prediction():
    """Fixtures to mock the EvalPrediction object passed to compute_metrics."""
    pred = MagicMock()
    # Mock logits: Shape (2, 3, 5) -> 2 samples, 3 timesteps, 5 vocabulary tokens
    pred.predictions = np.array([
        [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]],
        [[0.2, 0.2, 0.6], [0.1, 0.7, 0.2]]
    ])
    # Mock label IDs, including the loss masking token -100
    pred.label_ids = np.array([
        [1, -100],
        [2, 1]
    ])
    return pred


@patch("src.metrics.asr_metric.wer")
@patch("src.metrics.asr_metric.cer")
def test_compute_metrics_success(mock_cer, mock_wer, mock_processor, mock_eval_prediction):
    """Test that compute_metrics correctly processes inputs and returns scores."""
    # Arrange: Mock the return values for WER and CER compute methods
    mock_wer.compute.return_value = 0.12
    mock_cer.compute.return_value = 0.05

    # Act: Instantiate the curried function and call it
    compute_metrics_fn = create_metric(mock_processor)
    result = compute_metrics_fn(mock_eval_prediction)

    # Assert: Verify the final dictionary structure and values
    assert result == {"wer": 0.12, "cer": 0.05}

    # Assert: Verify -100 in label_ids was correctly replaced by pad_token_id (0)
    expected_labels = np.array([
        [1, 0],
        [2, 1]
    ])
    np.testing.assert_array_equal(mock_eval_prediction.label_ids, expected_labels)

    # Assert: Check if batch_decode was called with correct arguments
    # argmax of pred.predictions should yield [[1, 0], [2, 1]]
    expected_pred_ids = np.array([[1, 0], [2, 1]])
    
    # We capture the actual call arguments to verify the numpy arrays match
    pred_decode_call_args = mock_processor.batch_decode.call_args_list[0][0][0]
    np.testing.assert_array_equal(pred_decode_call_args, expected_pred_ids)
    
    # Assert: Ensure CER and WER compute methods were called with strings
    mock_wer.compute.assert_called_once_with(
        predictions=["decoded_string_0", "decoded_string_1"],
        references=["decoded_string_0", "decoded_string_1"]
    )
    mock_cer.compute.assert_called_once_with(
        predictions=["decoded_string_0", "decoded_string_1"],
        references=["decoded_string_0", "decoded_string_1"]
    )


@patch("src.metrics.asr_metric.wer")
@patch("src.metrics.asr_metric.cer")
def test_create_metric_returns_callable(mock_cer, mock_wer, mock_processor):
    """Test that create_metric successfully returns a curried callable function."""
    compute_metrics_fn = create_metric(mock_processor)
    assert callable(compute_metrics_fn)
    assert compute_metrics_fn.__name__ == "compute_metrics"
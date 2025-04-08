"""
Evaluation Module for Music Transcription System

This module provides functionality for evaluating the music transcription model.
It includes metrics for onset detection, pitch detection, note-level F1 score,
and other evaluation metrics.

Functions:
    evaluate_model: Evaluate model on test dataset
    calculate_metrics: Calculate evaluation metrics
    visualize_results: Visualize evaluation results
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import pretty_midi
import mir_eval
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from model.architecture import MusicTranscriptionModel
from preprocessing.dataset_preprocessor import create_datasets, create_test_transform
from inference.transcriber import MusicTranscriber


def calculate_note_metrics(
    ref_notes: List[Dict[str, Any]],
    est_notes: List[Dict[str, Any]],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2
) -> Dict[str, float]:
    """
    Calculate note-level evaluation metrics using mir_eval.
    
    Args:
        ref_notes (List[Dict[str, Any]]): Reference notes (ground truth)
        est_notes (List[Dict[str, Any]]): Estimated notes (predictions)
        onset_tolerance (float): Tolerance for onset detection in seconds
        offset_ratio (float): Ratio of the reference note duration used for offset tolerance
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Convert notes to mir_eval format
    ref_intervals = np.array([[note['onset_time'], note['offset_time']] for note in ref_notes])
    ref_pitches = np.array([note['pitch'] for note in ref_notes])
    
    est_intervals = np.array([[note['onset_time'], note['offset_time']] for note in est_notes])
    est_pitches = np.array([note['pitch'] for note in est_notes])
    
    # Handle empty arrays
    if len(ref_notes) == 0 or len(est_notes) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
    
    # Calculate metrics
    precision, recall, f1_score, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio
    )
    
    # Calculate accuracy
    if len(est_notes) > 0:
        accuracy = f1_score  # Use F1 score as accuracy for note detection
    else:
        accuracy = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def calculate_onset_metrics(
    ref_onsets: np.ndarray,
    est_onsets: np.ndarray,
    tolerance: float = 0.05
) -> Dict[str, float]:
    """
    Calculate onset detection metrics.
    
    Args:
        ref_onsets (np.ndarray): Reference onset times (ground truth)
        est_onsets (np.ndarray): Estimated onset times (predictions)
        tolerance (float): Tolerance for onset detection in seconds
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Handle empty arrays
    if len(ref_onsets) == 0 or len(est_onsets) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Calculate metrics
    precision, recall, f1_score = mir_eval.onset.f_measure(
        ref_onsets,
        est_onsets,
        window=tolerance
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def calculate_pitch_metrics(
    ref_pitches: np.ndarray,
    est_pitches: np.ndarray
) -> Dict[str, float]:
    """
    Calculate pitch detection metrics.
    
    Args:
        ref_pitches (np.ndarray): Reference pitches (ground truth)
        est_pitches (np.ndarray): Estimated pitches (predictions)
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Handle empty arrays
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
    
    # Find common indices
    common_indices = np.intersect1d(
        np.arange(len(ref_pitches)),
        np.arange(len(est_pitches))
    )
    
    if len(common_indices) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
    
    # Filter pitches
    ref_filtered = ref_pitches[common_indices]
    est_filtered = est_pitches[common_indices]
    
    # Calculate metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        ref_filtered == est_filtered,
        np.ones_like(ref_filtered),
        average='binary'
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(ref_filtered, est_filtered)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def calculate_tempo_metrics(
    ref_tempo: float,
    est_tempo: float,
    tolerance: float = 0.08
) -> Dict[str, float]:
    """
    Calculate tempo detection metrics.
    
    Args:
        ref_tempo (float): Reference tempo (ground truth)
        est_tempo (float): Estimated tempo (prediction)
        tolerance (float): Tolerance for tempo detection as a ratio
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Calculate relative error
    rel_error = abs(ref_tempo - est_tempo) / ref_tempo
    
    # Check if within tolerance
    is_correct = rel_error <= tolerance
    
    # Calculate accuracy
    accuracy = 1.0 if is_correct else 0.0
    
    return {
        'accuracy': accuracy,
        'relative_error': rel_error
    }


def evaluate_model(
    model: MusicTranscriptionModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = None,
    onset_threshold: float = 0.5,
    pitch_threshold: float = 0.5,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    Args:
        model (MusicTranscriptionModel): Music transcription model
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (torch.device, optional): Device to use for inference
        onset_threshold (float): Threshold for onset detection
        pitch_threshold (float): Threshold for pitch detection
        onset_tolerance (float): Tolerance for onset detection in seconds
        offset_ratio (float): Ratio of the reference note duration used for offset tolerance
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Initialize metrics
    all_metrics = {
        'note': {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        },
        'onset': {
            'precision': [],
            'recall': [],
            'f1_score': []
        },
        'pitch': {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        },
        'tempo': {
            'accuracy': [],
            'relative_error': []
        }
    }
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract features and targets
            features = batch['melspectrogram']
            
            # Forward pass
            predictions = model.predict(features, onset_threshold=onset_threshold)
            
            # Process each sample in the batch
            for i in range(features.size(0)):
                # Extract predictions
                onsets = predictions['onsets'][i].cpu().numpy()
                pitch_probs = predictions['pitch_probs'][i].cpu().numpy()
                durations = predictions['durations'][i].cpu().numpy()
                
                # Extract targets
                onset_target = batch['onset_target'][i].cpu().numpy()
                pitch_target = batch['pitch_target'][i].cpu().numpy()
                duration_target = batch['duration_target'][i].cpu().numpy()
                
                # Convert frame-level predictions to note events
                est_notes = []
                onset_frames = np.where(onsets > onset_threshold)[0]
                
                for frame in onset_frames:
                    # Get pitch probabilities for this frame
                    pitch_prob = pitch_probs[frame]
                    
                    # Find pitches above threshold
                    pitch_indices = np.where(pitch_prob > pitch_threshold)[0]
                    
                    # Get duration for this frame (in frames)
                    duration_frames = durations[frame]
                    
                    # Convert to seconds
                    duration_seconds = duration_frames * 0.01  # Assuming 10ms per frame
                    
                    # Calculate onset time
                    onset_time = frame * 0.01  # Assuming 10ms per frame
                    
                    # Create note events for each detected pitch
                    for pitch in pitch_indices:
                        note = {
                            'pitch': int(pitch),
                            'onset_time': onset_time,
                            'duration_time': duration_seconds,
                            'offset_time': onset_time + duration_seconds
                        }
                        est_notes.append(note)
                
                # Convert frame-level targets to note events
                ref_notes = []
                ref_onset_frames = np.where(onset_target > 0.5)[0]
                
                for frame in ref_onset_frames:
                    # Get pitches for this frame
                    pitches = np.where(pitch_target[frame] > 0.5)[0]
                    
                    # Get duration for this frame (in frames)
                    duration_frames = duration_target[frame]
                    
                    # Convert to seconds
                    duration_seconds = duration_frames * 0.01  # Assuming 10ms per frame
                    
                    # Calculate onset time
                    onset_time = frame * 0.01  # Assuming 10ms per frame
                    
                    # Create note events for each pitch
                    for pitch in pitches:
                        note = {
                            'pitch': int(pitch),
                            'onset_time': onset_time,
                            'duration_time': duration_seconds,
                            'offset_time': onset_time + duration_seconds
                        }
                        ref_notes.append(note)
                
                # Calculate note-level metrics
                note_metrics = calculate_note_metrics(
                    ref_notes,
                    est_notes,
                    onset_tolerance=onset_tolerance,
                    offset_ratio=offset_ratio
                )
                
                # Extract onset times
                ref_onset_times = np.array([note['onset_time'] for note in ref_notes])
                est_onset_times = np.array([note['onset_time'] for note in est_notes])
                
                # Calculate onset metrics
                onset_metrics = calculate_onset_metrics(
                    ref_onset_times,
                    est_onset_times,
                    tolerance=onset_tolerance
                )
                
                # Extract pitches
                ref_pitches = np.array([note['pitch'] for note in ref_notes])
                est_pitches = np.array([note['pitch'] for note in est_notes])
                
                # Calculate pitch metrics
                pitch_metrics = calculate_pitch_metrics(ref_pitches, est_pitches)
                
                # Calculate tempo metrics (if available)
                if 'tempo' in predictions and 'tempo_targets' in batch:
                    est_tempo = predictions['tempo'][i].cpu().numpy()
                    ref_tempo = batch['tempo_targets'][i].cpu().numpy()
                    
                    tempo_metrics = calculate_tempo_metrics(ref_tempo, est_tempo)
                    
                    # Update tempo metrics
                    for key, value in tempo_metrics.items():
                        all_metrics['tempo'][key].append(value)
                
                # Update metrics
                for key, value in note_metrics.items():
                    all_metrics['note'][key].append(value)
                
                for key, value in onset_metrics.items():
                    all_metrics['onset'][key].append(value)
                
                for key, value in pitch_metrics.items():
                    all_metrics['pitch'][key].append(value)
    
    # Calculate average metrics
    avg_metrics = {}
    
    for category, metrics in all_metrics.items():
        avg_metrics[category] = {}
        
        for key, values in metrics.items():
            if values:
                avg_metrics[category][key] = np.mean(values)
            else:
                avg_metrics[category][key] = 0.0
    
    return {
        'detailed': all_metrics,
        'average': avg_metrics
    }


def evaluate_transcriber(
    transcriber: MusicTranscriber,
    test_files: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluate transcriber on test files.
    
    Args:
        transcriber (MusicTranscriber): Music transcriber
        test_files (List[Dict[str, Any]]): List of test files with ground truth
        output_dir (str, optional): Directory to save evaluation results
        onset_tolerance (float): Tolerance for onset detection in seconds
        offset_ratio (float): Ratio of the reference note duration used for offset tolerance
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics
    """
    # Initialize metrics
    all_metrics = {
        'note': {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        },
        'onset': {
            'precision': [],
            'recall': [],
            'f1_score': []
        },
        'pitch': {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        },
        'tempo': {
            'accuracy': [],
            'relative_error': []
        }
    }
    
    # Process each test file
    for file_idx, file_data in enumerate(tqdm(test_files, desc="Evaluating")):
        # Extract file paths
        audio_path = file_data['audio_path']
        midi_path = file_data['midi_path']
        
        # Transcribe audio
        result = transcriber.transcribe_file(audio_path)
        est_notes = result['notes']
        est_tempo = result['tempo']
        
        # Load ground truth MIDI
        ref_midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Extract ground truth notes
        ref_notes = []
        for instrument in ref_midi.instruments:
            for note in instrument.notes:
                ref_notes.append({
                    'pitch': note.pitch,
                    'onset_time': note.start,
                    'offset_time': note.end,
                    'duration_time': note.end - note.start
                })
        
        # Get ground truth tempo
        ref_tempo = ref_midi.get_tempo_changes()[1][0] if ref_midi.get_tempo_changes()[1].size > 0 else 120.0
        
        # Calculate note-level metrics
        note_metrics = calculate_note_metrics(
            ref_notes,
            est_notes,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio
        )
        
        # Extract onset times
        ref_onset_times = np.array([note['onset_time'] for note in ref_notes])
        est_onset_times = np.array([note['onset_time'] for note in est_notes])
        
        # Calculate onset metrics
        onset_metrics = calculate_onset_metrics(
            ref_onset_times,
            est_onset_times,
            tolerance=onset_tolerance
        )
        
        # Extract pitches
        ref_pitches = np.array([note['pitch'] for note in ref_notes])
        est_pitches = np.array([note['pitch'] for note in est_notes])
        
        # Calculate pitch metrics
        pitch_metrics = calculate_pitch_metrics(ref_pitches, est_pitches)
        
        # Calculate tempo metrics
        tempo_metrics = calculate_tempo_metrics(ref_tempo, est_tempo)
        
        # Update metrics
        for key, value in note_metrics.items():
            all_metrics['note'][key].append(value)
        
        for key, value in onset_metrics.items():
            all_metrics['onset'][key].append(value)
        
        for key, value in pitch_metrics.items():
            all_metrics['pitch'][key].append(value)
        
        for key, value in tempo_metrics.items():
            all_metrics['tempo'][key].append(value)
        
        # Save visualization if output directory is provided
        if output_dir is not None:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load audio for visualization
            audio_data, _ = transcriber.audio_loader.load_file(audio_path)
            
            # Visualize transcription
            vis_output = os.path.join(output_dir, f"transcription_{file_idx}.png")
            transcriber.visualize_transcription(est_notes, audio_data, vis_output)
            
            # Save MIDI
            midi_output = os.path.join(output_dir, f"transcription_{file_idx}.mid")
            transcriber.save_midi(result['midi_data'], midi_output)
    
    # Calculate average metrics
    avg_metrics = {}
    
    for category, metrics in all_metrics.items():
        avg_metrics[category] = {}
        
        for key, values in metrics.items():
            if values:
                avg_metrics[category][key] = np.mean(values)
            else:
                avg_metrics[category][key] = 0.0
    
    # Save metrics if output directory is provided
    if output_dir is not None:
        metrics_output = os.path.join(output_dir, "evaluation_metrics.json")
        
        with open(metrics_output, 'w') as f:
            json.dump({
                'average': avg_metrics,
                'detailed': all_metrics
            }, f, indent=2)
    
    return {
        'detailed': all_metrics,
        'average': avg_metrics
    }


def visualize_results(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None
) -> None:
    """
    Visualize evaluation results.
    
    Args:
        metrics (Dict[str, Any]): Dictionary of evaluation metrics
        output_path (str, optional): Path to save visualization
    """
    # Extract average metrics
    avg_metrics = metrics['average']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot note metrics
    ax = axes[0, 0]
    note_metrics = avg_metrics['note']
    x = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    y = [note_metrics['precision'], note_metrics['recall'], note_metrics['f1_score'], note_metrics['accuracy']]
    ax.bar(x, y)
    ax.set_ylim(0, 1)
    ax.set_title('Note-level Metrics')
    ax.set_ylabel('Score')
    
    # Plot onset metrics
    ax = axes[0, 1]
    onset_metrics = avg_metrics['onset']
    x = ['Precision', 'Recall', 'F1 Score']
    y = [onset_metrics['precision'], onset_metrics['recall'], onset_metrics['f1_score']]
    ax.bar(x, y)
    ax.set_ylim(0, 1)
    ax.set_title('Onset Detection Metrics')
    ax.set_ylabel('Score')
    
    # Plot pitch metrics
    ax = axes[1, 0]
    pitch_metrics = avg_metrics['pitch']
    x = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    y = [pitch_metrics['precision'], pitch_metrics['recall'], pitch_metrics['f1_score'], pitch_metrics['accuracy']]
    ax.bar(x, y)
    ax.set_ylim(0, 1)
    ax.set_title('Pitch Detection Metrics')
    ax.set_ylabel('Score')
    
    # Plot tempo metrics
    ax = axes[1, 1]
    tempo_metrics = avg_metrics['tempo']
    x = ['Accuracy', '1 - Relative Error']
    y = [tempo_metrics['accuracy'], 1 - tempo_metrics['relative_error']]
    ax.bar(x, y)
    ax.set_ylim(0, 1)
    ax.set_title('Tempo Detection Metrics')
    ax.set_ylabel('Score')
    
    # Add overall title
    fig.suptitle('Music Transcription Evaluation Results', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save or show figure
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    from inference.transcriber import load_model, MusicTranscriber
    
    parser = argparse.ArgumentParser(description="Evaluate music transcription model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--preprocessed_dir", type=str, help="Directory containing preprocessed data")
    parser.add_argument("--test_files", type=str, help="JSON file with test file paths")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--onset_threshold", type=float, default=0.5, help="Threshold for onset detection")
    parser.add_argument("--pitch_threshold", type=float, default=0.5, help="Threshold for pitch detection")
    parser.add_argument("--onset_tolerance", type=float, default=0.05, help="Tolerance for onset detection in seconds")
    parser.add_argument("--offset_ratio", type=float, default=0.2, help="Ratio of the reference note duration used for offset tolerance")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    if args.preprocessed_dir:
        # Create test dataloader
        dataloaders = create_datasets(
            preprocessed_dir=args.preprocessed_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transforms={
                'test': create_test_transform()
            }
        )
        
        test_loader = dataloaders['test']
        
        # Evaluate model on test dataset
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            onset_threshold=args.onset_threshold,
            pitch_threshold=args.pitch_threshold,
            onset_tolerance=args.onset_tolerance,
            offset_ratio=args.offset_ratio
        )
        
        # Save metrics
        metrics_output = os.path.join(args.output_dir, "evaluation_metrics.json")
        
        with open(metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Visualize results
        vis_output = os.path.join(args.output_dir, "evaluation_results.png")
        visualize_results(metrics, vis_output)
        
        print("Evaluation completed!")
        print(f"Metrics saved to {metrics_output}")
        print(f"Visualization saved to {vis_output}")
    
    elif args.test_files:
        # Create transcriber
        transcriber = MusicTranscriber(
            model=model,
            sample_rate=args.sample_rate,
            onset_threshold=args.onset_threshold,
            pitch_threshold=args.pitch_threshold
        )
        
        # Load test files
        with open(args.test_files, 'r') as f:
            test_files = json.load(f)
        
        # Evaluate transcriber on test files
        metrics = evaluate_transcriber(
            transcriber=transcriber,
            test_files=test_files,
            output_dir=args.output_dir,
            onset_tolerance=args.onset_tolerance,
            offset_ratio=args.offset_ratio
        )
        
        # Visualize results
        vis_output = os.path.join(args.output_dir, "evaluation_results.png")
        visualize_results(metrics, vis_output)
        
        print("Evaluation completed!")
        print(f"Results saved to {args.output_dir}")
    
    else:
        print("Either --preprocessed_dir or --test_files must be provided")

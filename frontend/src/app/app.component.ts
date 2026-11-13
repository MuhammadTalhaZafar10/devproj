import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PredictionApiService } from './prediction-api.service';

interface ModelResult {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  macroF1: number;
  weightedF1: number;
  trainingTime: number;
}

interface PredictionResult {
  prediction: string;
  inference_mode: string;
  model_version: string;
  probabilities: Record<string, number>;
  conformal_prediction_set: string[];
  note?: string;
}

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  backendState = 'Checking inference service...';
  backendAvailable = false;
  predictionLoading = false;
  liveResult: PredictionResult | null = {
    prediction: 'High',
    inference_mode: 'demo',
    model_version: 'histgb-seed42',
    probabilities: { Low: 0.0009, Medium: 0.0001, High: 0.999 },
    conformal_prediction_set: ['High'],
    note: 'Preview from the validated seed-42 sample. Start Flask for live inference.'
  };
  liveError = '';
  private samplePayload: Record<string, number> = {};
  editableValues: Record<string, number> = {
    hour: 12,
    day: 15,
    month: 6,
    energy_lag_1: 0,
    energy_lag_24: 0,
    rolling_mean_3: 0,
    rolling_mean_24: 0
  };
  readonly editableFields = [
    { key: 'hour', label: 'Hour of day', min: 0, max: 23, step: 1 },
    { key: 'day', label: 'Day of month', min: 1, max: 31, step: 1 },
    { key: 'month', label: 'Month', min: 1, max: 12, step: 1 },
    { key: 'energy_lag_1', label: 'Energy lag · 1 hour', min: -100000, max: 100000, step: 0.01 },
    { key: 'energy_lag_24', label: 'Energy lag · 24 hours', min: -100000, max: 100000, step: 0.01 },
    { key: 'rolling_mean_3', label: 'Rolling mean · 3 hours', min: -100000, max: 100000, step: 0.01 },
    { key: 'rolling_mean_24', label: 'Rolling mean · 24 hours', min: -100000, max: 100000, step: 0.01 }
  ];

  constructor(private readonly predictionApi: PredictionApiService) {}

  ngOnInit(): void {
    this.predictionApi.health().subscribe({
      next: (health) => {
        this.backendAvailable = true;
        this.backendState = `Connected · ${health.inference_mode} · ${health.model_version}`;
        this.loadSamplePayload(true);
      },
      error: () => {
        this.backendState = 'Preview mode · Flask service not connected';
        this.loadSamplePayload(false);
      }
    });
  }

  private loadSamplePayload(runPrediction: boolean): void {
    fetch('sample_request.json')
      .then((response) => response.json())
      .then((payload: Record<string, number>) => {
        this.samplePayload = payload;
        for (const field of this.editableFields) {
          this.editableValues[field.key] = payload[field.key] ?? this.editableValues[field.key];
        }
        if (runPrediction) {
          this.submitPrediction();
        }
      })
      .catch(() => this.liveError = 'Sample payload could not be loaded.');
  }

  submitPrediction(): void {
    if (!this.backendAvailable) {
      this.liveError = 'Flask is not connected. Run "python backend/app.py" and refresh this page locally.';
      return;
    }
    this.predictionLoading = true;
    this.liveError = '';
    const payload = { ...this.samplePayload, ...this.editableValues };
    this.predictionApi.predict(payload).subscribe({
      next: (result) => {
        this.liveResult = result;
        this.predictionLoading = false;
      },
      error: () => {
        this.liveError = 'The Flask API rejected this input. Check the values and try again.';
        this.predictionLoading = false;
      }
    });
  }

  readonly modelResults: ModelResult[] = [
    {
      name: 'LightGBM',
      accuracy: 0.9212,
      precision: 0.8968,
      recall: 0.9098,
      macroF1: 0.903,
      weightedF1: 0.9217,
      trainingTime: 12.036
    },
    {
      name: 'HistGradientBoosting',
      accuracy: 0.9168,
      precision: 0.9001,
      recall: 0.9051,
      macroF1: 0.8985,
      weightedF1: 0.9168,
      trainingTime: 18.42
    },
    {
      name: 'Random Forest',
      accuracy: 0.8931,
      precision: 0.878,
      recall: 0.8538,
      macroF1: 0.8646,
      weightedF1: 0.8917,
      trainingTime: 15.871
    }
  ];

  readonly researchMetrics = [
    { value: '91.68%', label: 'test accuracy', detail: 'trained model' },
    { value: '91.68%', label: 'conformal coverage', detail: '90% nominal target' },
    { value: '0.1176', label: 'ordinal MAE', detail: 'Low < Medium < High' },
    { value: '300', label: 'input features', detail: 'schema-controlled' }
  ];

  readonly workflow = [
    { number: '01', title: 'BDG2 data', text: '100,000 labeled energy rows from the project dataset.' },
    { number: '02', title: 'Ensemble baseline', text: 'Gradient boosting adapted from the recent ensemble/XAI methodology.' },
    { number: '03', title: 'Calibration split', text: 'A separate 16,000-row split estimates prediction uncertainty.' },
    { number: '04', title: 'Prediction set', text: 'The API returns probabilities and an uncertainty-aware class set.' }
  ];

  readonly highlights = [
    { label: 'Final observations', value: '100,000', detail: 'Selected real rows' },
    { label: 'Processed features', value: '300', detail: 'After preprocessing' },
    { label: 'Training rows', value: '80,000', detail: '80% of the split' },
    { label: 'Testing rows', value: '20,000', detail: '20% of the split' }
  ];

  readonly topFeatures = [
    { name: 'rolling_mean_3', importance: 3194 },
    { name: 'energy_lag_1', importance: 1932 },
    { name: 'rolling_std_24', importance: 1101 },
    { name: 'rolling_mean_24', importance: 583 },
    { name: 'hour', importance: 519 },
    { name: 'hour_sin', importance: 485 },
    { name: 'building_id_kaggle', importance: 481 },
    { name: 'energy_lag_24', importance: 479 }
  ];

  readonly maxImportance = 3194;
}

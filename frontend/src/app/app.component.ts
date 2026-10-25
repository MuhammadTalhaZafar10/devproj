import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

interface ModelResult {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  macroF1: number;
  weightedF1: number;
  trainingTime: number;
}

@Component({
  selector: 'app-root',
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
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
      name: 'Random Forest',
      accuracy: 0.8931,
      precision: 0.878,
      recall: 0.8538,
      macroF1: 0.8646,
      weightedF1: 0.8917,
      trainingTime: 15.871
    }
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

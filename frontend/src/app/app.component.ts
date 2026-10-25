import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { catchError, of } from 'rxjs';
import { PredictionApiService } from './prediction-api.service';

@Component({
  selector: 'app-root',
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  readonly navItems = [
    { label: 'Dashboard', icon: '⌂' },
    { label: 'Threat Scanner', icon: '⌁' },
    { label: 'Analysis History', icon: '◷' },
    { label: 'Documentation', icon: '?' }
  ];
  schemaLoaded = false;
  healthMessage = 'Checking service...';
  healthOnline = false;
  inferenceMode = '—';
  featureCount = 0;
  activeNav = 'Dashboard';
  isScanning = false;
  resultMessage = '';
  resultLabel = '';
  errorMessage = '';
  validationError = '';
  featureNames: string[] = [];
  payloadControl = new FormControl<string>('', { nonNullable: true });

  constructor(private readonly api: PredictionApiService) {}

  ngOnInit(): void {
    this.bootstrapSchema();
    this.checkHealth();
  }

  private async bootstrapSchema(): Promise<void> {
    try {
      const [schemaRes, sampleRes] = await Promise.all([
        fetch('feature_schema.json'),
        fetch('sample_request.json')
      ]);
      const schemaData = await schemaRes.json() as { features: string[] };
      const sampleData = await sampleRes.json() as Record<string, number>;

      this.featureNames = schemaData.features;
      this.featureCount = this.featureNames.length;
      this.payloadControl.setValue(JSON.stringify(sampleData, null, 2));
      this.schemaLoaded = true;
    } catch {
      this.validationError = 'Unable to load local schema/sample assets.';
    }
  }

  private checkHealth(): void {
    this.api.health().pipe(
      catchError(() => of(null))
    ).subscribe((health) => {
      if (!health) {
        this.healthMessage = 'Backend unavailable';
        this.healthOnline = false;
        return;
      }
      this.healthMessage = 'System operational';
      this.healthOnline = true;
      this.inferenceMode = health.inference_mode;
      this.featureCount = health.feature_count;
    });
  }

  private parseAndValidatePayload(raw: string): Record<string, number> | null {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      this.validationError = 'Payload must be valid JSON.';
      return null;
    }

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      this.validationError = 'Payload must be a JSON object.';
      return null;
    }

    const payload = parsed as Record<string, unknown>;
    const keys = Object.keys(payload);
    const missing = this.featureNames.filter((name) => !keys.includes(name));
    const extra = keys.filter((name) => !this.featureNames.includes(name));

    if (missing.length) {
      this.validationError = `Missing required fields (${missing.length}). First: ${missing.slice(0, 5).join(', ')}`;
      return null;
    }
    if (extra.length) {
      this.validationError = `Unexpected fields (${extra.length}). First: ${extra.slice(0, 5).join(', ')}`;
      return null;
    }

    const normalized: Record<string, number> = {};
    for (const feature of this.featureNames) {
      const value = Number(payload[feature]);
      if (!Number.isFinite(value)) {
        this.validationError = `Field '${feature}' must be a finite number.`;
        return null;
      }
      normalized[feature] = value;
    }
    this.validationError = '';
    return normalized;
  }

  submit(): void {
    this.resultMessage = '';
    this.errorMessage = '';
    const payload = this.parseAndValidatePayload(this.payloadControl.value);
    if (!payload) {
      return;
    }

    this.isScanning = true;
    this.api.predict(payload).subscribe({
      next: (response) => {
        this.isScanning = false;
        this.resultLabel = response.prediction;
        this.resultMessage = response.note || `Analysis completed using ${response.inference_mode} inference.`;
      },
      error: (err) => {
        this.isScanning = false;
        const apiMessage = err?.error?.error;
        this.errorMessage = apiMessage || 'Backend request failed. Ensure Flask is running locally.';
      }
    });
  }

  loadSample(): void {
    this.validationError = '';
    fetch('sample_request.json')
      .then((response) => response.json() as Promise<Record<string, number>>)
      .then((sample) => this.payloadControl.setValue(JSON.stringify(sample, null, 2)))
      .catch(() => {
        this.validationError = 'Unable to load the sample payload.';
      });
  }

  selectNav(label: string): void {
    this.activeNav = label;
  }
}

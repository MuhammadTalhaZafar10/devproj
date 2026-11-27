import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, ValidatorFn } from '@angular/forms';
import { PredictionApiService } from './prediction-api.service';
import { ClassName, Prediction } from './models';

export interface ClsSchema {
  features: string[];
  classes?: ClassName[];
  model?: string;
  defaults?: Record<string, number>;
  ranges?: Record<string, [number, number]>;
}

interface Field {
  name: string;
  label: string;
}

interface FieldGroup {
  title: string;
  fields: Field[];
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const CLASSES: ClassName[] = ['Low', 'Medium', 'High'];

/** Plain-words label for a feature name. */
export function featureLabel(name: string): string {
  let m = /^hour_(\d+)$/.exec(name);
  if (m) return `hour ${m[1].padStart(2, '0')}:00`;
  m = /^dow_(\d)$/.exec(name);
  if (m) return `${DAYS[+m[1]] ?? name} level`;
  m = /^q(\d+)$/.exec(name);
  if (m) return `percentile ${+m[1]}`;
  m = /^period_(\w+)$/.exec(name);
  if (m) return `${m[1]} period`;
  m = /^acf_lag(\d+)$/.exec(name);
  if (m) return `autocorrelation, lag ${m[1]} h`;
  if (name === 'temp_corr') return 'temperature correlation';
  return name.replace(/_/g, ' ');
}

/** Required, finite, and inside [lo, hi] when a range is known. */
function numberValidator(range?: [number, number]): ValidatorFn {
  return (c: AbstractControl): ValidationErrors | null => {
    const v = c.value;
    if (v === null || v === undefined || v === '') return { required: true };
    if (typeof v !== 'number' || !Number.isFinite(v)) return { number: true };
    if (range && (v < range[0] || v > range[1])) return { range: true };
    return null;
  };
}

@Component({
  selector: 'app-feature-form',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './feature-form.component.html',
})
export class FeatureFormComponent implements OnInit {
  private readonly http = inject(HttpClient);
  private readonly api = inject(PredictionApiService);

  readonly schema = signal<ClsSchema | null>(null);
  readonly form = signal<FormGroup | null>(null);
  readonly loadFailed = signal(false);
  readonly loading = signal(false);
  readonly prediction = signal<Prediction | null>(null);
  readonly errorText = signal('');
  readonly classes = CLASSES;

  readonly groups = computed<FieldGroup[]>(() => {
    const s = this.schema();
    if (!s) return [];
    const field = (name: string): Field => ({ name, label: featureLabel(name) });
    const hourly = s.features.filter((n) => n.startsWith('hour_'));
    const weekday = s.features.filter((n) => n.startsWith('dow_'));
    const weather: string[] = s.features.filter((n) => n === 'temp_corr');
    const shape = s.features.filter((n) => !hourly.includes(n) && !weekday.includes(n) && !weather.includes(n));
    return [
      { title: 'Hourly profile', fields: hourly.map(field) },
      { title: 'Weekday profile', fields: weekday.map(field) },
      { title: 'Shape of the load', fields: shape.map(field) },
      { title: 'Weather', fields: weather.map(field) },
    ].filter((g) => g.fields.length > 0);
  });

  ngOnInit(): void {
    this.http.get<ClsSchema>('cls_schema.json').subscribe({
      next: (s) => {
        this.schema.set(s);
        const controls: Record<string, FormControl<number | null>> = {};
        for (const name of s.features) {
          controls[name] = new FormControl<number | null>(null, numberValidator(s.ranges?.[name]));
        }
        this.form.set(new FormGroup(controls));
      },
      error: () => this.loadFailed.set(true),
    });
  }

  /** True when the field should show its error: after the user has touched or edited it. */
  showError(name: string): boolean {
    const c = this.form()?.get(name);
    return !!c && c.invalid && (c.touched || c.dirty);
  }

  errorMessage(name: string): string {
    const range = this.schema()?.ranges?.[name];
    if (this.form()?.get(name)?.hasError('range') && range) {
      return `Enter a number between ${range[0]} and ${range[1]}`;
    }
    return 'Enter a number';
  }

  useTypicalValues(): void {
    const defaults = this.schema()?.defaults;
    const f = this.form();
    if (!defaults || !f) return;
    for (const name of Object.keys(f.controls)) {
      if (name in defaults) f.get(name)!.setValue(defaults[name]);
    }
    f.markAsDirty();
  }

  hasDefaults(): boolean {
    return Object.keys(this.schema()?.defaults ?? {}).length > 0;
  }

  classify(): void {
    const f = this.form();
    if (!f || f.invalid || this.loading()) return;
    const features = f.getRawValue() as Record<string, number>;
    this.loading.set(true);
    this.errorText.set('');
    this.api.predictFeatures(features).subscribe({
      next: (p) => {
        this.prediction.set(p);
        this.loading.set(false);
      },
      error: (e) => {
        this.prediction.set(null);
        this.loading.set(false);
        this.errorText.set(
          e.status === 0
            ? "The backend isn't running. Start it with python -m backend.app, then try again."
            : (e.error?.error ?? 'The API rejected these values.'),
        );
      },
    });
  }

  pct(v: number): string {
    return `${Math.round(v * 100)}%`;
  }
}

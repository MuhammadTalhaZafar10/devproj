import { Component, computed, effect, inject, input, signal } from '@angular/core';
import { PredictionApiService } from './prediction-api.service';
import { ClassName, Example, Prediction } from './models';

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const W = 720;
const H = 220;
const PAD = { l: 44, r: 8, t: 10, b: 24 };

@Component({
  selector: 'app-trace-panel',
  standalone: true,
  templateUrl: './trace-panel.component.html',
})
export class TracePanelComponent {
  private readonly api = inject(PredictionApiService);

  readonly examples = input.required<Example[]>();
  /** Share of single weeks from unseen buildings the served model gets right (0 to 1). */
  readonly weekAccuracy = input<number | null>(null);
  readonly selected = signal(0);
  readonly prediction = signal<Prediction | null>(null);
  readonly status = signal<'idle' | 'loading' | 'offline' | 'error'>('idle');
  readonly errorText = signal('');
  readonly W = W;
  readonly H = H;
  readonly pad = PAD;

  readonly current = computed(() => this.examples()[this.selected()]);

  readonly yMax = computed(() => {
    const vals = this.current().hourly_kwh.filter((v): v is number => v !== null);
    const m = Math.max(...vals, 1);
    const pow = Math.pow(10, Math.floor(Math.log10(m)));
    return Math.ceil(m / pow) * pow;
  });

  readonly path = computed(() => {
    const xs = (i: number) => PAD.l + (i / 167) * (W - PAD.l - PAD.r);
    const ys = (v: number) => PAD.t + (1 - v / this.yMax()) * (H - PAD.t - PAD.b);
    let d = '';
    let pen = false;
    this.current().hourly_kwh.forEach((v, i) => {
      if (v === null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  });

  readonly dayBands = computed(() => {
    const start = new Date(this.current().week_start.replace(' ', 'T')).getDay();
    const dayW = (W - PAD.l - PAD.r) / 7;
    return Array.from({ length: 7 }, (_, k) => ({
      x: PAD.l + k * dayW,
      w: dayW,
      label: DAYS[(start + 6 + k) % 7],
      weekend: [5, 6].includes((start + 6 + k) % 7),
    }));
  });

  readonly yTicks = computed(() => {
    const top = this.yMax();
    return [0, 0.5, 1].map((f) => ({
      label: Math.round(top * f).toLocaleString(),
      y: PAD.t + (1 - f) * (H - PAD.t - PAD.b),
    }));
  });

  readonly classes: ClassName[] = ['Low', 'Medium', 'High'];

  constructor() {
    effect(() => {
      this.selected();
      this.prediction.set(null);
      this.status.set('idle');
    });
  }

  pick(i: number): void {
    this.selected.set(i);
  }

  classify(): void {
    const ex = this.current();
    this.status.set('loading');
    this.api
      .predict({
        hourly_kwh: ex.hourly_kwh,
        air_temperature: ex.air_temperature,
        week_start: ex.week_start.slice(0, 10),
      })
      .subscribe({
        next: (p) => {
          this.prediction.set(p);
          this.status.set('idle');
        },
        error: (e) => {
          this.prediction.set(null);
          if (e.status === 0) {
            this.status.set('offline');
          } else {
            this.errorText.set(e.error?.error ?? 'The API rejected this request.');
            this.status.set('error');
          }
        },
      });
  }

  pct(v: number): string {
    return `${Math.round(v * 100)}%`;
  }

  shortName(id: string): string {
    return id.split('_').slice(1).join(' ');
  }
}

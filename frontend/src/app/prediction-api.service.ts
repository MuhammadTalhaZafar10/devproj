import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, map, of, shareReplay, switchMap } from 'rxjs';
import { Example, Health, Novelty, Prediction, Results } from './models';

const LOCAL_API = 'http://127.0.0.1:5000';

export interface PredictRequest {
  hourly_kwh: (number | null)[];
  air_temperature?: number[];
  week_start?: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionApiService {
  private readonly http = inject(HttpClient);

  /**
   * Where the live backend lives. Read from config.json next to the page so the host can be
   * changed after deployment without rebuilding. The dev server (port 4200) uses its /api proxy.
   */
  private readonly base$: Observable<string> = this.http
    .get<{ apiBase?: string }>('config.json')
    .pipe(
      map((c) => (c.apiBase ?? LOCAL_API).replace(/\/+$/, '')),
      catchError(() => of(LOCAL_API)),
      map((configured) => (this.onDevServer() ? '/api' : configured)),
      shareReplay(1),
    );

  private onDevServer(): boolean {
    return typeof location !== 'undefined' && location.port === '4200' &&
      ['localhost', '127.0.0.1'].includes(location.hostname);
  }

  /** Static files shipped with the site, so the page renders without a backend. */
  results(): Observable<Results> {
    return this.http.get<Results>('classification_results.json');
  }

  examples(): Observable<Example[]> {
    return this.http.get<Example[]>('cls_examples.json');
  }

  novelty(): Observable<Novelty | null> {
    return this.http.get<Novelty>('cls_novelty_results.json').pipe(catchError(() => of(null)));
  }

  /** Live backend, optional. */
  health(): Observable<Health> {
    return this.base$.pipe(switchMap((b) => this.http.get<Health>(`${b}/health`)));
  }

  predict(body: PredictRequest): Observable<Prediction> {
    return this.base$.pipe(switchMap((b) => this.http.post<Prediction>(`${b}/predict`, body)));
  }

  /** Manual path: one number per model feature, posted as a flat object. */
  predictFeatures(features: Record<string, number>): Observable<Prediction> {
    return this.base$.pipe(switchMap((b) => this.http.post<Prediction>(`${b}/predict`, features)));
  }
}

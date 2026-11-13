import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

const configuredApiUrl = (globalThis as typeof globalThis & { __BDG2_API_URL__?: string }).__BDG2_API_URL__;
const isPages = globalThis.location?.hostname.endsWith('.github.io') ?? false;
const apiBaseUrl = configuredApiUrl || (isPages ? 'https://bdg2-uncertainty-api.onrender.com' : '/api');

@Injectable({ providedIn: 'root' })
export class PredictionApiService {
  constructor(private readonly http: HttpClient) {}

  health() {
    return this.http.get<{
      status: string;
      inference_mode: string;
      feature_count: number;
      model_version: string;
    }>(`${apiBaseUrl}/health`);
  }

  predict(payload: Record<string, number>) {
    return this.http.post<{
      prediction: string;
      inference_mode: string;
      model_version: string;
      probabilities: Record<string, number>;
      conformal_prediction_set: string[];
      note?: string;
    }>(`${apiBaseUrl}/predict`, payload);
  }
}

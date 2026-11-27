import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ClsSchema, FeatureFormComponent } from './feature-form.component';

const schema: ClsSchema = {
  features: ['hour_00', 'dow_0', 'temp_corr'],
  classes: ['Low', 'Medium', 'High'],
  model: 'LightGBM',
  defaults: { hour_00: 0.8, dow_0: 1.1, temp_corr: 0.25 },
  ranges: { hour_00: [0, 24], dow_0: [0, 5], temp_corr: [-1, 1] },
};

describe('FeatureFormComponent', () => {
  let fixture: ComponentFixture<FeatureFormComponent>;
  let http: HttpTestingController;
  let el: HTMLElement;

  const input = (name: string) => el.querySelector<HTMLInputElement>(`#ff-${name}`)!;
  const type = (name: string, value: string) => {
    const i = input(name);
    i.value = value;
    i.dispatchEvent(new Event('input'));
    fixture.detectChanges();
  };
  const button = (text: string) =>
    Array.from(el.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.includes(text))!;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [FeatureFormComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(FeatureFormComponent);
    el = fixture.nativeElement;
    fixture.detectChanges();
    http.expectOne('cls_schema.json').flush(schema);
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  it('renders one number input per feature, grouped under headings', () => {
    expect(el.querySelectorAll('input[type="number"]').length).toBe(3);
    const legends = Array.from(el.querySelectorAll('legend')).map((l) => l.textContent?.trim());
    expect(legends).toEqual(['Hourly profile', 'Weekday profile', 'Weather']);
    expect(el.querySelector('label[for="ff-dow_0"]')?.textContent).toContain('Mon level');
  });

  it('is invalid when a value is outside its range', () => {
    type('hour_00', '30');
    type('dow_0', '1');
    type('temp_corr', '0');
    expect(button('Classify').disabled).toBeTrue();
    expect(el.textContent).toContain('Enter a number between 0 and 24');

    type('hour_00', '12');
    expect(button('Classify').disabled).toBeFalse();
  });

  it('fills the defaults with "Use typical values"', () => {
    expect(button('Classify').disabled).toBeTrue();
    button('Use typical values').click();
    fixture.detectChanges();
    expect(input('hour_00').value).toBe('0.8');
    expect(input('dow_0').value).toBe('1.1');
    expect(input('temp_corr').value).toBe('0.25');
    expect(button('Classify').disabled).toBeFalse();
  });

  it('posts the flat feature object to /predict and shows the result', () => {
    button('Use typical values').click();
    fixture.detectChanges();
    button('Classify').click();

    http.expectOne('config.json').flush({ apiBase: 'http://api.test' });
    const req = http.expectOne('http://api.test/predict');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ hour_00: 0.8, dow_0: 1.1, temp_corr: 0.25 });
    req.flush({ prediction: 'High', probabilities: { Low: 0.1, Medium: 0.2, High: 0.7 }, model: 'LightGBM' });
    fixture.detectChanges();

    expect(el.querySelector('.class-pill')?.textContent).toContain('High');
    expect(el.textContent).toContain('High 70%');
  });
});

# Studio głosu — lokalny model na GPU

**Linux / wynajęta maszyna 2× RTX 3090: [pełna instrukcja i komendy](RENTAL.md).**

W panelu można dodawać MP4: automatycznie powstaje MP3 do pobrania oraz WAV do treningu. Samodzielna konwersja pliku lub całego folderu: `python -m studio.media SCIEZKA --output converted`. Windows: przeciągnij film lub folder na `MP4_NA_MP3.bat`. Wymagany FFmpeg w PATH.

Uruchom **START.bat**, a następnie otwórz http://127.0.0.1:7860.

1. Dodaj wiele nagrań jednego głosu. Obsługiwane WAV, FLAC, MP3 i OGG, zależnie od kodeków libsndfile.
2. Kliknij „Przygotuj próbki”. Program dzieli nagrania na fragmenty do 12 s i rozpoznaje polski tekst przez Whisper na GPU. Sprawdź transkrypcje; wyłącz błędne fragmenty. Cięcia mogą wypaść w środku słowa — takie fragmenty najlepiej wyłączyć.
3. Wytrenuj model. Kolejne uruchomienie treningu wczytuje poprzednie wagi adapterów, stan optymalizatora i skalowania gradientów. Dodanie próbek nie zeruje modelu. Każdy trening obejmuje wszystkie aktywne fragmenty, także wcześniejsze.
4. Generuj mowę z zapisanego modelu. Nie trzeba ponownie przesyłać próbek ani trenować.

## Co jest zapisywane

- `data/samples`: nagrania wejściowe mono i metadane; duplikaty są rozpoznawane po zawartości pliku.
- `data/clips`: fragmenty i poprawialne transkrypcje.
- `data/features`: jednorazowo obliczone tokeny dźwięku i cechy głosu.
- `data/models/latest.pt`: rzeczywiste wytrenowane adaptery LoRA, optymalizator, liczba kroków, uśredniona reprezentacja głosu. Do działania potrzebna jest również baza KittenTTS2 z lokalnego cache Hugging Face.
- `data/models/reference.wav`: stała referencja dekodera dźwięku. Nie jest ponownie przesyłana przez użytkownika.
- `data/outputs`: gotowe WAV.
- `data/jobs`: statusy i dzienniki zadań.
- `archive`: zachowane wcześniejsze skrypty i ich wyniki. Oryginalne WAV są zachowane w `archive/originals` oraz zaimportowane do panelu.

## Trening i ograniczenia

To własny eksperymentalny trainer LoRA dla KittenTTS2 0.9.4, nie oficjalny trainer producenta. Uczy projekcje Q/V uwagi na tokenach prawdziwych nagrań z transkrypcją; baza i dekoder pozostają zamrożone. Generator używa tych samych adapterów i tego samego formatu warunkowania głosem co trening. Jakość trzeba oceniać odsłuchem. Kilkadziesiąt sekund wystarczy do testu działania, nie do oceny jakości głosu.

BF16, CUDA, akumulacja gradientów i opcjonalny checkpointing aktywacji. Model wymaga zakresu BF16 — test rzeczywistego fragmentu wykazał przepełnienie FP16 w projekcji uwagi warstwy 23. RTX 2070 nie ma natywnego przyspieszenia BF16, więc obliczenia są wolniejsze niż na RTX 3090. Whisper korzysta z FP16. Domyślny batch na RTX 2070 to jeden fragment; panel pozwala ustawić rzeczywisty batch na GPU oraz akumulację osobno. Profil 2×3090 zaczyna od batcha 4 na kartę. Nie wymuszamy sztucznego obciążenia GPU: przygotowanie danych, zapis i generowanie autoregresyjne nie utrzymują stałych 100%. Zamknięcie innych programów używających GPU zwiększa dostępną pamięć. Błąd braku VRAM jest pokazywany w panelu.

Zapis atomowy po każdej epoce, co 25 kroków oraz po przycisku zatrzymania (na granicy aktualizacji). Kontynuacja zachowuje uczenie, ale nie odtwarza dokładnie pozycji i kolejności próbek w przerwanej epoce. Nagłe wyłączenie zachowuje ostatni ukończony zapis. Jedno zadanie korzysta z GPU w danej chwili. Modele są zwalniane po zadaniu; generowanie ładuje wagi ponownie, ale ich nie trenuje.

Od 10 fragmentów jest wydzielane około 10% na walidację. Mniejszy zestaw nie ma miarodajnego pomiaru generalizacji. Pierwsze uruchomienie może pobrać wagi z Hugging Face; nagrania nie są wysyłane. Zachowaj lokalny cache modeli, aby nie pobierać ich ponownie.

Powered by KittenML. Baza i jej pochodne podlegają licencji producenta: https://kittenml.com/license/.

## Sprawdzenie działania — 8 listopada 2026

Na RTX 2070 przygotowano 18 fragmentów z dwóch istniejących nagrań (łącznie około 3,6 minuty). Trening ukończył 1 epokę i zapisał 5 aktualizacji. Ponowne uruchomienie wczytało zapis, wykonało aktualizacje do kroku 7 i zapisało stan po zatrzymaniu. Generowanie z tego modelu utworzyło WAV 24 kHz o długości 3,36 s, bez zmiany liczby kroków treningu. Zaobserwowane obciążenie podczas treningu: 97–98%, pamięć około 6,5 GB. To test działania, nie potwierdzenie jakości ani podobieństwa głosu.

Testy podstawowe: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
Rzeczywisty test gradientów na GPU: `.venv\Scripts\python.exe -m tests.gpu_probe`.
Pełny test kontynuacji i generowania: `.venv\Scripts\python.exe -m tests.check_workflow` (wymaga uruchomionego panelu i zapisanego modelu; wykonuje dalszy trening i zapisuje wynik).

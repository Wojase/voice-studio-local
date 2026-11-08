# Wynajęta maszyna: Linux + 2× RTX 3090

Kod przygotowano do dwóch procesów CUDA/NCCL, po jednym na kartę. Każdy proces ma własną kopię bazy i przetwarza inne próbki. Gradienty adapterów są synchronizowane przed aktualizacją. 24 GB + 24 GB nie staje się jedną pulą 48 GB. Generowanie pojedynczego nagrania i przygotowanie próbek używają jednej karty; dwie pracują przy treningu. Nie ma gwarancji stałych 100%: zależy to od danych, batcha, dysku i połączenia kart.

Wymagania: Ubuntu 24.04, Python 3.12+, sterownik NVIDIA zgodny z CUDA 12.8, dwie widoczne karty, zalecane 64 GB RAM oraz co najmniej 40 GB wolnego miejsca plus przestrzeń na dane. Nie musisz instalować pełnego CUDA Toolkit; biblioteki dostarcza PyTorch. Setup instaluje pakiety systemowe i środowisko Python. Przy pierwszym użyciu pobiera wagi bazowe i Whisper. Nie publikuje nagrań.

## Instalacja na serwerze

```bash
git clone https://github.com/Wojase/voice-studio-local.git
cd voice-studio-local
bash scripts/setup-linux.sh
source .venv/bin/activate
nvidia-smi
python -c 'import torch; print(torch.__version__,torch.cuda.device_count()); assert torch.cuda.device_count()==2'
```

W osobnym terminalu na swoim komputerze otwórz tunel (podmień `USER`, `HOST` i port SSH):

```bash
ssh -p 22 -L 7860:127.0.0.1:7860 USER@HOST
```

Na serwerze uruchom panel w tmux, aby przetrwał rozłączenie SSH:

```bash
sudo apt-get install -y tmux
tmux new -s voice
bash scripts/start-3090.sh
```

Na komputerze otwórz http://127.0.0.1:7860. Odłącz tmux: Ctrl+B, potem D. Powrót: `tmux attach -t voice`. Panel słucha tylko na loopback; nie wystawiaj portu 7860 publicznie. W kontenerze uruchamiaj tunel w tej samej przestrzeni sieciowej co panel albo skonfiguruj forward dostawcy do loopback kontenera.

## Przeniesienie obecnego głosu

Prześlij cały folder `data` do katalogu projektu na serwerze przy zatrzymanych zadaniach. Przykład PowerShell z katalogu lokalnego projektu:

```powershell
scp -P 22 -r .\data USER@HOST:/SCIEZKA/voice-studio-local/
```

Zachowaj `samples`, `clips`, `features`, cały `models` (w tym `latest.pt`, `reference.wav`, `split.json`) i `outputs`. Folder `jobs` to tylko historia. Nie przenoś Windowsowej `.venv`: środowisko Linux tworzy setup. Modele bazowe pobiorą się ponownie na serwerze. Po przeniesieniu sprawdź tekst próbek i kontynuuj trening. Zatrzymaj zadania przed kopiowaniem danych w dowolną stronę.

## Konwersja i import wielu nagrań

```bash
source .venv/bin/activate
# Jeden plik albo folder przetwarzany rekurencyjnie; oryginały pozostają.
python -m studio.media /sciezka/do/filmow --output /sciezka/do/mp3
# Import bezpośrednio z oryginalnych filmów / audio, bez stratnego etapu MP3.
python -m studio.importer /sciezka/do/filmow
python -m studio.cli prepare
```

MP4 dodane przez panel automatycznie tworzy MP3 w `data/converted` oraz WAV do treningu. CLI pomija już istniejące MP3 i pokazuje błędy plików bez ścieżki audio. MP3 jest skompresowany stratnie; do treningu używamy WAV wyodrębnionego z oryginału.

## Trening bez panelu — obie karty

Nie uruchamiaj CLI równocześnie z zadaniem w panelu. Używają tych samych danych i zapisów.

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node=2 --module studio.cli train \
  --epochs 10 --batch-size 4 --accumulation 2 --lr 0.00005
```

To efektywnie do 16 fragmentów na aktualizację: 4 × 2 akumulacje × 2 GPU. Ostatnia grupa jest dopełniana próbkami, aby oba procesy miały tyle samo aktualizacji. Rozpocznij od batcha 4, sprawdź VRAM, potem spróbuj 6 lub 8. Po braku pamięci wróć do mniejszego batcha; stan z ostatniego zapisu pozostaje. Większy batch zmienia zachowanie uczenia, więc sprawdzaj walidację i odsłuch. Przy dostatecznej pamięci można wyłączyć checkpointing aktywacji (`export STUDIO_CHECKPOINTING=0`), co ogranicza ponowne obliczenia, ale zużywa więcej VRAM.

Przerwij Ctrl+C; zapis następuje po bieżącej aktualizacji. Następne identyczne polecenie wczytuje wagi i optymalizator. Zachowuje dotychczasową liczbę ukończonych epok, lecz przerwaną epokę zaczyna ponownie. `latest.pt` jest stanem do kontynuacji, `best.pt` zachowuje najniższy zaobserwowany błąd walidacji. Generowanie domyślnie używa `latest.pt`.

```bash
# Jedna karta do generowania z gotowego modelu.
CUDA_VISIBLE_DEVICES=0 python -m studio.cli generate --text 'Dzień dobry, to mój model głosu.'
# Monitoring podczas treningu, w drugim terminalu.
watch -n 1 nvidia-smi
# Testy podstawowe.
python -m unittest discover -s tests -v
```

Przed długim treningiem wykonaj jedną epokę na małym zestawie. Tryb dwóch RTX 3090 wymaga pomiaru i sprawdzenia na docelowym serwerze; lokalnie zweryfikowano synchronizację dwóch procesów CPU, konwersję FFmpeg i trening jednego GPU. Rozwiązanie nadal jest eksperymentalnym trainerem KittenTTS2, a nie oficjalnym narzędziem producenta.

Opis mechanizmu komunikacji: [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html), [torchrun](https://docs.pytorch.org/docs/main/elastic/run.html).

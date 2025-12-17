"""
Script complet pour enrichir un dataset Uber avec des données météo
Utilisation: python uber_weather_analysis.py
"""

import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np

# Installer geopy si nécessaire: pip install geopy
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
except ImportError:
    print("❌ Erreur: geopy n'est pas installé")
    print("📦 Installez avec: pip install geopy")
    exit()


class WeatherEnricher:
    """
    Enrichit un dataset Uber avec des données météorologiques
    """

    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.geolocator = Nominatim(user_agent="uber_weather_ml_project_2024")
        self.cache = {}
        self.location_cache = {}

    def categorize_weather(self, weather_code):
        """Catégorise selon WMO Weather codes"""
        weather_mapping = {
            0: "Clear",
            1: "Partly Cloudy", 2: "Partly Cloudy", 3: "Partly Cloudy",
            45: "Foggy", 48: "Foggy",
            51: "Light Rain", 53: "Moderate Rain", 55: "Heavy Rain",
            61: "Light Rain", 63: "Moderate Rain", 65: "Heavy Rain",
            71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
            80: "Rain Showers", 81: "Rain Showers", 82: "Heavy Rain Showers",
            85: "Snow", 86: "Snow",
            95: "Thunderstorm", 96: "Thunderstorm", 99: "Heavy Thunderstorm"
        }
        return weather_mapping.get(weather_code, "Cloudy")

    def clean_location_name(self, location):
        """Nettoie le nom de localisation"""
        if pd.isna(location):
            return None

        parts = [p.strip() for p in str(location).split(',')]

        if len(parts) >= 2:
            cleaned = f"{parts[0]}, {parts[1]}, India"
        elif len(parts) == 1:
            cleaned = f"{parts[0]}, India"
        else:
            cleaned = str(location)

        return cleaned

    def geocode_location(self, location):
        """Convertit un nom de lieu en coordonnées (lat, lon)"""
        if pd.isna(location):
            return None, None

        if location in self.location_cache:
            return self.location_cache[location]

        cleaned_location = self.clean_location_name(location)

        try:
            loc_result = self.geolocator.geocode(cleaned_location, timeout=10)

            if loc_result:
                lat, lon = loc_result.latitude, loc_result.longitude
                self.location_cache[location] = (lat, lon)
                time.sleep(1.1)  # Rate limiting
                return lat, lon
            else:
                print(f"⚠️ Géocodage échoué pour: {cleaned_location}")
                self.location_cache[location] = (None, None)
                return None, None

        except GeocoderTimedOut:
            print(f"⏱️ Timeout pour: {cleaned_location}")
            return None, None
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None, None

    def get_weather(self, lat, lon, date_str, hour):
        """Récupère les conditions météo"""
        cache_key = f"{lat}_{lon}_{date_str}_{hour}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": ["temperature_2m", "precipitation", "weathercode", "rain"],
                "timezone": "auto"
            }

            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            hourly = data['hourly']

            if hour < len(hourly['weathercode']):
                result = {
                    'condition': self.categorize_weather(hourly['weathercode'][hour]),
                    'temperature': hourly['temperature_2m'][hour],
                    'is_rainy': int(hourly['weathercode'][hour] in [
                        51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99
                    ])
                }

            else:
                result = None

            self.cache[cache_key] = result
            time.sleep(0.05)

            return result

        except Exception as e:
            print(f"❌ Erreur API météo: {e}")
            return None

    def enrich_dataset(self, df, sample_size=None):
        """Enrichit le dataset avec les données météo"""
        print("🚀 Début de l'enrichissement météo")
        print(f"📊 Taille du dataset: {len(df)} lignes")

        df_work = df.copy()

        if sample_size:
            df_work = df_work.sample(n=min(sample_size, len(df_work)), random_state=42)
            print(f"🎲 Échantillon de {len(df_work)} lignes")

        # Parser dates
        print("\n📅 Parsing des dates...")
        df_work['datetime'] = pd.to_datetime(df_work['Date'] + ' ' + df_work['Time'])
        df_work['date_only'] = df_work['datetime'].dt.date.astype(str)
        df_work['hour'] = df_work['datetime'].dt.hour

        # Géocoder localisations uniques
        print(f"\n🗺️ Géocodage des localisations...")
        unique_locations = df_work['Pickup Location'].dropna().unique()
        print(f"   {len(unique_locations)} localisations uniques")

        location_coords = {}
        for i, loc in enumerate(unique_locations, 1):
            lat, lon = self.geocode_location(loc)
            location_coords[loc] = (lat, lon)

            if i % 5 == 0 or i == len(unique_locations):
                print(f"   Progression: {i}/{len(unique_locations)} ({i / len(unique_locations) * 100:.1f}%)")

        # Appliquer coordonnées
        print("\n🎯 Application des coordonnées...")
        df_work['latitude'] = df_work['Pickup Location'].map(lambda x: location_coords.get(x, (None, None))[0])
        df_work['longitude'] = df_work['Pickup Location'].map(lambda x: location_coords.get(x, (None, None))[1])

        geocoded = df_work['latitude'].notna().sum()
        print(f"   ✅ {geocoded}/{len(df_work)} lignes géocodées ({geocoded / len(df_work) * 100:.1f}%)")

        # Récupérer météo
        print("\n🌦️ Récupération des données météo...")
        weather_results = []

        for idx, row in df_work.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                weather = self.get_weather(
                    row['latitude'],
                    row['longitude'],
                    row['date_only'],
                    row['hour']
                )

                if weather:
                    weather_results.append(weather)
                else:
                    weather_results.append(self._empty_weather())
            else:
                weather_results.append(self._empty_weather())

            if len(weather_results) % 50 == 0:
                print(
                    f"   Traité: {len(weather_results)}/{len(df_work)} ({len(weather_results) / len(df_work) * 100:.1f}%)")

        # Créer DataFrame météo
        weather_df = pd.DataFrame(weather_results)
        result_df = pd.concat([df_work.reset_index(drop=True), weather_df], axis=1)

        # Stats
        print("\n" + "=" * 60)
        print("✅ ENRICHISSEMENT TERMINÉ !")
        print("=" * 60)
        print(f"\n📊 Distribution des conditions météo:")
        print(result_df['condition'].value_counts())
        print(f"\n🌡️ Température moyenne: {result_df['temperature'].mean():.1f}°C")


        return result_df

    def _empty_weather(self):
        return {
            'condition': 'Unknown',
            'temperature': None,
            'is_rainy': 0
        }


# ============== PROGRAMME PRINCIPAL ==============

def main():
    """Fonction principale"""

    print("=" * 70)
    print("   ENRICHISSEMENT MÉTÉO - DATASET UBER")
    print("=" * 70)

    # 1. Charger le dataset
    print("\n📂 Chargement du dataset...")

    # MODIFIEZ LE NOM DE VOTRE FICHIER ICI
    filename = 'uber_rides_data.csv'

    try:
        df = pd.read_csv(filename)
        print(f"✅ Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier '{filename}' n'existe pas")
        print("💡 Assurez-vous que le fichier CSV est dans le même dossier que ce script")
        return
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return

    # Vérifier les colonnes nécessaires
    required_cols = ['Date', 'Time', 'Pickup Location']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        print(f"📋 Colonnes disponibles: {df.columns.tolist()}")
        return

    print("\n🔍 Aperçu des données:")
    print(df[['Date', 'Time', 'Pickup Location']].head())

    # 2. Créer l'enrichisseur
    enricher = WeatherEnricher()

    # 3. Demander la taille de l'échantillon
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print("Pour tester, commencez avec 50-100 lignes")
    print("Pour le dataset complet, entrez 0")

    while True:
        try:
            sample = input("\nNombre de lignes à traiter (0 = tout, ex: 100): ").strip()
            sample_size = int(sample) if sample else 100

            if sample_size == 0:
                sample_size = None
                print(f"⚠️ Vous allez traiter TOUT le dataset ({len(df)} lignes)")
                print(f"⏱️ Temps estimé: {len(df) * 0.05 / 60:.0f} minutes")
                confirm = input("Continuer? (oui/non): ").strip().lower()
                if confirm != 'oui':
                    sample_size = 100
                    print("→ Utilisation de 100 lignes par défaut")
            break
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")

    # 4. Enrichir le dataset
    print("\n" + "=" * 70)
    print("TRAITEMENT")
    print("=" * 70)

    df_enriched = enricher.enrich_dataset(df, sample_size=sample_size)

    # 5. Sauvegarder
    output_file = 'uber_with_weather.csv'
    df_enriched.to_csv(output_file, index=False)
    print(f"\n💾 Fichier sauvegardé: {output_file}")

    # 6. Analyse rapide
    if 'Booking Value' in df_enriched.columns:
        print("\n" + "=" * 70)
        print("ANALYSE RAPIDE - Impact Météo sur Prix")
        print("=" * 70)

        rainy_rides = df_enriched[df_enriched['is_rainy'] == 1]
        clear_rides = df_enriched[df_enriched['is_rainy'] == 0]

        if len(rainy_rides) > 0 and len(clear_rides) > 0:
            avg_rain = rainy_rides['Booking Value'].mean()
            avg_clear = clear_rides['Booking Value'].mean()

            print(f"\n💧 Prix moyen (pluie):  ₹{avg_rain:.2f}")
            print(f"☀️  Prix moyen (beau):   ₹{avg_clear:.2f}")

            if avg_clear > 0:
                increase = ((avg_rain - avg_clear) / avg_clear) * 100
                print(f"📊 Différence:          {increase:+.1f}%")

                if increase > 0:
                    print(f"\n✅ Les prix augmentent de {increase:.1f}% lors de pluie!")
                else:
                    print(f"\n📉 Les prix diminuent de {abs(increase):.1f}% lors de pluie")

    print("\n" + "=" * 70)
    print("✅ TERMINÉ - Dataset prêt pour Machine Learning!")
    print("=" * 70)
    print(f"\n📄 Fichier généré: {output_file}")
    print(f"📊 Lignes traitées: {len(df_enriched)}")
    print(f"🆕 Nouvelles colonnes: condition, temperature, precipitation, is_rainy, etc.")

    return df_enriched


# ============== LANCER LE SCRIPT ==============

if __name__ == "__main__":
    try:
        df_result = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback

        traceback.print_exc()
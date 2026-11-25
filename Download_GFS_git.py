#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:46:15 2026

@author: wbarros


Baixa vento (u10m, v10m) do modelo Eta 1km (CPTEC) para o ano de 2025 em uma grade
regular de 13x13 pontos.

Cada arquivo .grb é amostrado pelo vizinho mais próximo nos 169 pontos da
grade. A saída final é um NetCDF (time, latitude, longitude) no mesmo formato do


Antes de rodar, ajuste as variáveis para a sua seção CONFIG abaixo.

Recomendação: LER O LEIA-ME

"""

import os
import glob
import time
import pickle
import requests
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm


##################################################################
# CONFIG — AJUSTE AQUI
##################################################################

# Pasta onde ficam os arquivos temporários, o checkpoint e o NetCDF final.
# Troque pelo caminho da sua pasta.
PASTA_SAIDA = "/caminho/da/sua/pasta/vento_eta_2025"

# Período a baixar.
# Troque pelo SEU PERÍODO.
DATA_INICIO = datetime(2025, 1, 1)
DATA_FIM = datetime(2025, 12, 31)

# Grade de destino.
# Troque pelas suas coordenadas.
LAT_GRID = np.arange(-21.5, -24.5 - 1e-9, -0.25)
LON_GRID = np.arange(-44.5, -41.5 + 1e-9, 0.25)

# URL base do servidor CPTEC.
URL_BASE = "https://dataserver.cptec.inpe.br/dataserver_modelos/eta/rjsp_01km/brutos"

# Ajustes de download.
#Para evitar BLOW-OUT
PAUSA_ENTRE_DOWNLOADS = 0.5
TIMEOUT = 60
MAX_TENTATIVAS = 3

##################################################################


ARQUIVO_TEMP = os.path.join(PASTA_SAIDA, "temp_eta.grb")
ARQUIVO_CHECKPOINT = os.path.join(PASTA_SAIDA, "checkpoint_vento_eta_grade.pkl")
ARQUIVO_SAIDA_NC = os.path.join(PASTA_SAIDA, "vento_eta_2025_grade.nc")


def monta_url(dia, hora):
    """URL do .grb para a rodada 00Z de `dia`, passo `hora`."""
    data_str = dia.strftime("%Y%m%d")
    init_str = f"{data_str}00"
    valid = dia + timedelta(hours=hora)
    valid_str = valid.strftime("%Y%m%d%H")
    return (
        f"{URL_BASE}/{dia.year:04d}/{dia.month:02d}/{dia.day:02d}/00/"
        f"eta_01km_{init_str}+{valid_str}.grb"
    )


def baixa_arquivo(url, destino):
    """Baixa `url` para `destino`. Retorna True em sucesso."""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200 and len(r.content) > 1_000_000:
                with open(destino, "wb") as f:
                    f.write(r.content)
                return True
            print(f"  status={r.status_code} tamanho={len(r.content)} tentativa={tentativa}")
        except Exception as e:
            print(f"  tentativa {tentativa} falhou: {e}")
        time.sleep(2)
    return False


def extrai_vento_grade(caminho_grb):
    """Extrai u10m/v10m nos pontos da grade por vizinho mais próximo."""
    shape = (len(LAT_GRID), len(LON_GRID))
    ds = None
    try:
        ds = xr.open_dataset(
            caminho_grb,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
        )
        sub = ds.sel(latitude=LAT_GRID, longitude=LON_GRID, method="nearest")
        u10 = np.asarray(sub.u10m.values, dtype="float32").reshape(shape)
        v10 = np.asarray(sub.v10m.values, dtype="float32").reshape(shape)
        return u10, v10
    except Exception as e:
        print(f"  falha na extração: {e}")
        nan_grid = np.full(shape, np.nan, dtype="float32")
        return nan_grid, nan_grid
    finally:
        if ds is not None:
            ds.close()


def carrega_checkpoint():
    if os.path.exists(ARQUIVO_CHECKPOINT):
        with open(ARQUIVO_CHECKPOINT, "rb") as f:
            return pickle.load(f)
    return {}


def salva_checkpoint(resultados):
    with open(ARQUIVO_CHECKPOINT, "wb") as f:
        pickle.dump(resultados, f)


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    resultados = carrega_checkpoint()
    print(f"Checkpoint: {len(resultados)} horas já processadas.")
    print(
        f"Grade: {len(LAT_GRID)}x{len(LON_GRID)} pontos "
        f"(lat {LAT_GRID[0]} a {LAT_GRID[-1]}, lon {LON_GRID[0]} a {LON_GRID[-1]})"
    )

    total_horas = int((DATA_FIM - DATA_INICIO).total_seconds() / 3600) + 24
    todos_ts = [DATA_INICIO + timedelta(hours=h) for h in range(total_horas)]
    faltam = [t for t in todos_ts if t not in resultados]
    print(f"Faltam {len(faltam)} de {total_horas} horas.")

    barra = tqdm(faltam, desc="Eta 1km", unit="h")
    desde_checkpoint = 0

    for ts in barra:
        dia_base = ts.replace(hour=0, minute=0, second=0)
        url = monta_url(dia_base, ts.hour)

        if baixa_arquivo(url, ARQUIVO_TEMP):
            u10_grid, v10_grid = extrai_vento_grade(ARQUIVO_TEMP)
        else:
            shape = (len(LAT_GRID), len(LON_GRID))
            u10_grid = np.full(shape, np.nan, dtype="float32")
            v10_grid = np.full(shape, np.nan, dtype="float32")

        resultados[ts] = (u10_grid, v10_grid)

        if os.path.exists(ARQUIVO_TEMP):
            os.remove(ARQUIVO_TEMP)
        for lixo in glob.glob(ARQUIVO_TEMP + "*.idx"):
            os.remove(lixo)

        pct = 100 * len(resultados) / total_horas
        barra.set_postfix({"total": f"{len(resultados)}/{total_horas}", "%": f"{pct:.1f}"})

        desde_checkpoint += 1
        if desde_checkpoint % 24 == 0:
            salva_checkpoint(resultados)

        time.sleep(PAUSA_ENTRE_DOWNLOADS)

    salva_checkpoint(resultados)

    print("Montando NetCDF...")
    tempos = sorted(resultados.keys())
    u10_stack = np.stack([resultados[t][0] for t in tempos], axis=0)
    v10_stack = np.stack([resultados[t][1] for t in tempos], axis=0)

    ds_final = xr.Dataset(
        {
            "u10m": (("time", "latitude", "longitude"), u10_stack),
            "v10m": (("time", "latitude", "longitude"), v10_stack),
        },
        coords={
            "time": pd.to_datetime(tempos),
            "latitude": LAT_GRID,
            "longitude": LON_GRID,
        },
        attrs={
            "description": (
                "Vento 10m do Eta 1km (CPTEC) amostrado na grade do ERA5 "
                "(13x13, 0.25 graus) por vizinho mais próximo."
            ),
        },
    )
    ds_final.to_netcdf(ARQUIVO_SAIDA_NC)
    print(f"Salvo: {ARQUIVO_SAIDA_NC}")


if __name__ == "__main__":
    main()
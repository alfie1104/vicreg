import chromadb
import pandas as pd

# Chroma DB 경로 설정
DB_PATH = "chroma_db"

# PersistentClient 설정
client = chromadb.PersistentClient(path=DB_PATH)

# 컬렉션 생성 또는 불러오기
collection = client.get_or_create_collection(
    name="layout_image",
    metadata={"hnsw:space":"cosine"} # default가 l2 로 설정된 distance 측정 함수를 cosine으로 변경
)


def save_embeded_image(data : pd.DataFrame):
    """
    DB_LCTN, ID_VT, HDDN_RVSN, FEATURE_0, ..., FEATURE_1023으로 구성된 pd.DataFrame을 Chroma DB에 저장
    """
    # Vector DB에 결과 기록
    embeddings = data.iloc[:, 3:].values.tolist() #  DB_LCTN, ID_VT, HDDN_RVSN을 제외한 나머지를 embeddings로 사용
    
    # DB_LCTN, ID_VT, HDDN_RVSN을 이용하여 ids 생성
    metadatas = data[['DB_LCTN', 'ID_VT', 'HDDN_RVSN']].to_dict('records')

    # ids 생성 (각 메타데이터 필드를 조합하여 유일한 ID 생성)
    ids = [
        f"{meta['DB_LCTN']}-{meta['ID_VT']}-{meta['HDDN_RVSN']}" for meta in metadatas
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Results saved to ChromaDB")
    



def search_by_metadata(db_lctn : str, id_vt :str, hddn_rvsn: int):
    """
    DB_LCTN, ID_VT, HDDN_RVSN을 이용한 임베딩값 검색
    """
    results = collection.query(
        query_texts=None,
        where={
            "DB_LCTN": db_lctn,
            "ID_VT":id_vt,
            "HDDN_RVSN":hddn_rvsn
        },
        n_results=1
    )

    return results

def search_similar_vt_image(query_embedding, tok_k=10, db_lctn: str = None, id_vt : str = None, hddn_rvsn : int = None):
    """
    1024차원으로 embedding된 이미지 벡터를 입력받아서 가장 유사한 top_k개 VT의 DB_LCTN, ID_VT, HDDN_RVSN 그리고 그때의 유사도를 반환한다.
    """
    if db_lctn != None and id_vt != None and hddn_rvsn != None:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=tok_k,
            where={
                # Chroma DB의 쿼리 필터링 방식은 현재 복합조건을 직접 지원하지 않음
                # 따라서 $or 연산자를 이용하여 세 조건이 모두 같은 경우만 제외하도록 구성하였음
                # DB_LCTN != db_lctn || ID_VT != id_vt || HDDN_RVSN != hddn_rvsn이 됨으로써 결과적으로 모두 같은 경우 false || false || false가 되어 제외되게 됨
                "$or": [
                    {"DB_LCTN": {"$ne": db_lctn}},  # DB_LCTN이 다름
                    {"ID_VT": {"$ne": id_vt}},      # ID_VT가 다름
                    {"HDDN_RVSN": {"$ne": int(hddn_rvsn)}}  # HDDN_RVSN이 다름 # hddn_rvsn이 float인지 int인지 표시하기 위해 int로 변환 적용(chroma db에서 str, int, float을 따로 관리함)
                ]
            }
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=tok_k
        )

    # ChromaDb의 쿼리 결과는 일반적으로 다음의 구조를 지님
    # ids : 쿼리 결과의 ID 목록
    # embeddings : 해당 ID의 임베딩 값
    # distances : 쿼리 벡터와 각 임베딩 간의 거리
    # metadatas : 각 ID에 대한 메타데이터

    # 결과에 유사도 점수 및 메타데이터 포함
    similar_images = []

    # Chroma DB는 여러개의 query를 한번에 처리할 수 있으므로 results가 항상 2차원 배열로 반환됨
    # 따라서 단일 쿼리를 보냈더라도 실제 결과값을 사용하기 위해서는 [0]이라는 인덱스를 붙여서 활용해야함
    for idx, score, metadata in zip(results["ids"][0], results['distances'][0], results["metadatas"][0]):        
        if isinstance(metadata, dict):
            similar_images.append({
                "DB_LCTN":metadata.get("DB_LCTN"),
                "ID_VT":metadata.get("ID_VT"),
                "HDDN_RVSN":metadata.get("HDDN_RVSN"),
                "SIMILARITY_SCORE": round(max(1- score,0),5)
            })
    
    return similar_images
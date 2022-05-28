
def make_specific_link(youtube_link):
    # youtube link 유효성 체크
    youtube_link = youtube_link.url
    if "youtu.be" not in youtube_link and "www.youtube.com" not in youtube_link:
        return False
    # 유효성 검사 완료
    # else:    
    # 상세 주소 가져오기
    specific_link = youtube_link.split('/')[-1]
    # 링크 그대로 가져왔을 때 후처리
    if "?v=" in specific_link:
        specific_link = specific_link.split('?v=')[1].split('&')[0]

    return specific_link

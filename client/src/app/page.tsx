// app/page.tsx
'use client';

import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';
import 'swiper/css/pagination';
import FileUploader from '../../src/components/FileUploader';

import { Pagination } from 'swiper/modules';

// Material UI 컴포넌트 가져오기
import { SpeedDial, SpeedDialAction, SpeedDialIcon } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import SearchIcon from '@mui/icons-material/Search';
import ShareIcon from '@mui/icons-material/Share';

type Action = {
  icon: JSX.Element;
  name: string;
};

const actions: Action[] = [
  { icon: <HomeIcon />, name: 'Home' },
  { icon: <SearchIcon />, name: 'Search' },
  { icon: <ShareIcon />, name: 'Share' },
];

export default function Home() {
  return (
    <div style={{ height: '100vh', position: 'relative' }}>
      {/* 전체 배경으로 Swiper 설정 */}
      <Swiper
        modules={[Pagination]} // pagination 모듈
        spaceBetween={0} // 슬라이드 간 간격 제거
        slidesPerView={1} // 한 번에 보일 슬라이드 개수
        centeredSlides={true} // 슬라이드가 항상 중앙에 위치
        pagination={{ clickable: true }} // 페이지네이션 사용
        style={{ width: '100vw', height: '100vh', position: 'absolute', top: 0, left: 0 }}
      >
        <SwiperSlide>
          <img
            src="/image1.jpg"
            alt="Slide 1"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} // 이미지가 화면 전체를 덮도록 설정
          />
        </SwiperSlide>
        <SwiperSlide>
          <img
            src="/image2.jpg"
            alt="Slide 2"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        </SwiperSlide>
        <SwiperSlide>
          <img
            src="/image3.jpg"
            alt="Slide 3"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        </SwiperSlide>
        <SwiperSlide>
          <img
            src="/image4.jpg"
            alt="Slide 4"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        </SwiperSlide>
      </Swiper>

      {/* Material UI SpeedDial */}
      <SpeedDial
        ariaLabel="SpeedDial basic example"
        sx={{ position: 'absolute', bottom: 16, right: 16 }}
        icon={<SpeedDialIcon />}
      >
        {actions.map((action) => (
          <SpeedDialAction
            key={action.name}
            icon={action.icon}
            tooltipTitle={action.name}
          />
        ))}
      </SpeedDial>

      <FileUploader />
    </div>
  );
}

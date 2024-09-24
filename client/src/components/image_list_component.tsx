'use client'

import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import Image from 'next/image';
import { Typography, Box } from '@mui/material';

interface ImageListComponentProps {
  imageDatas: ImageData[];
}

interface ImageData {
  img: string;
  title: string;
  text: string;
}

export const ImageListComponent: React.FC<ImageListComponentProps> = ({
  imageDatas,
}) => {
  const theme = useTheme();

  // 반응형 열 수 설정
  const isXl = useMediaQuery(theme.breakpoints.up('xl'));
  const isLg = useMediaQuery(theme.breakpoints.up('lg'));
  const isMd = useMediaQuery(theme.breakpoints.up('md'));
  const isSm = useMediaQuery(theme.breakpoints.up('sm'));

  let cols = 1;
  if (isXl) cols = 6;
  else if (isLg) cols = 5;
  else if (isMd) cols = 4;
  else if (isSm) cols = 3;
  else cols = 1;

  return (
    <ImageList variant="masonry" cols={cols} gap={8}>
      {imageDatas.map((item, idx) => (
        <ImageListItem
          key={idx}
          sx={{
            position: 'relative',
            cursor: 'pointer',
            transition: 'box-shadow 0.3s ease',
            '&:hover': {
              boxShadow: 6,
            },
            '&:hover .overlay': {
              opacity: 1,
            },
          }}
        >
          <Image
            src={item.img}
            alt={item.title}
            width={500}
            height={500}
            style={{ width: '100%', height: 'auto' }}
          />
          <Box
            className="overlay"
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              bgcolor: 'rgba(0, 0, 0, 0.5)',
              opacity: 0,
              transition: 'opacity 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Typography variant="h6" color="white" align="center" px={2}>
              {item.text}
            </Typography>
          </Box>
        </ImageListItem>
      ))}
    </ImageList>
  );
};

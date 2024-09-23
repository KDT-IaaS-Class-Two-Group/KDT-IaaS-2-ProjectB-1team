import { Card, CardBody, CardHeader } from "@nextui-org/card";
import ImageList from "@mui/material/ImageList";
import ImageListItem from "@mui/material/ImageListItem";
import { Black_Han_Sans } from '@next/font/google';

const black_han_sans = Black_Han_Sans({
  subsets: ['latin'],
  weight: "400" // 사용할 폰트 두께 설정
});

interface TextCardProps {
  head: string;
  body: string;
  imageDatas: ImageData[]
}

interface ImageData {
  img: string,
  title: string,
}

export const TextCard: React.FC<TextCardProps> = ({ head, body, imageDatas }) => {

  return (
    <Card isBlurred className="w-[600px] h-[500px]">
      <CardHeader className="flex flex-row justify-between">
        <div style={{ fontFamily: black_han_sans.style.fontFamily }} className="text-5xl">{head}</div>
      </CardHeader>
      <CardBody>
        <div className="w-2/3">{body}</div>
        <ImageList variant="masonry" cols={3} gap={8} className="mt-3">
          {imageDatas.map((item) => (
            <ImageListItem key={item.img}>
              <img
                srcSet={`${item.img}?w=248&fit=crop&auto=format&dpr=2 2x`}
                src={`${item.img}?w=248&fit=crop&auto=format`}
                alt={item.title}
                loading="lazy"
              />
            </ImageListItem>
          ))}
        </ImageList>
      </CardBody>
    </Card>
  );
};

import { Video as LucideVideo, LucideProps } from 'lucide-react';

const Video = ({ className, ...props }: LucideProps) => {
  return <LucideVideo className={className} {...props} />;
};

export default Video;
import { Video as LucideVideo, LucideProps } from 'lucide-react';

const Videos = ({ className, ...props }: LucideProps) => {
  return <LucideVideo className={className} {...props} />;
};

export default Videos;
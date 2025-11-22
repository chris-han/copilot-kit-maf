import { Box as LucideBox, LucideProps } from 'lucide-react';

const Box = ({ className, ...props }: LucideProps) => {
  return <LucideBox className={className} {...props} />;
};

export default Box;